""" Deep Sets
"""
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from fetcher import DATA_PATH
from fetcher import FILE_NAME
from fetcher import get_cards
from fetcher import get_removables
from fetcher import get_replacements
from fetcher import load_and_update_decklists


class SetsDataset(Dataset):
    """ A dataset class for Deep Sets.
    """

    def __init__(self, slots: list[dict[str, int]], vocab: list[str], vector_type: str = "count"):
        """ Initialize the dataset.

        :param slots: A list of slots, where each slot is a dictionary with element IDs as keys and counts as values.
        :param vector_type: The type of vector to use ("count" or "binary").
        """
        assert vector_type in ["count", "binary"], "vector_type must be either 'count' or 'binary'"

        self.slots = slots
        self.vocab = vocab
        self.vector_type = vector_type

    def __len__(self):
        return len(self.slots)

    def __getitem__(self, idx):
        slot = self.slots[idx]
        if self.vector_type == 'count':
            return torch.tensor([slot.get(w, 0) for w in self.vocab], dtype=torch.float32)

        return torch.tensor([min(1, slot.get(w, 0)) for w in self.vocab], dtype=torch.float32)


class SetsDataModule(pytorch_lightning.LightningDataModule):
    """ A data module class for Deep Sets.
    """

    def __init__(
            self,
            slots: list[dict[str, int]], vocab: list[str],
            val_split: float = 0.2, batch_size: int = 16, vector_type: str = "count"):
        super().__init__()
        assert 0 < val_split < 1, "val_split must be between 0 and 1"
        assert vector_type in ["count", "binary"], "vector_type must be either 'count' or 'binary'"

        self.val_dataset = None
        self.train_dataset = None

        self.slots = slots
        self.vocab = vocab
        self.val_split = val_split
        self.batch_size = batch_size
        self.vector_type = vector_type

    def setup(self, stage: str = None) -> None:
        """ Setup the data for the model.

        :param stage: The stage of the data preparation (not needed, default is None).
        """

        full_dataset = SetsDataset(self.slots, self.vocab, vector_type=self.vector_type)
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """ Get the training data loader.

        :return: The training data loader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=9,
                          persistent_workers=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        """ Get the validation data loader.

        :return: The validation data loader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=9, persistent_workers=True)


class DeepSetsModel(LightningModule):
    def __init__(self, vocab_size=300, embed_dim=128, hidden_dim=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # φ: element-wise encoder
        self.phi = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        # ρ: set-level encoder
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)  # reconstruct original vector
        )

    def forward(self, x):
        # x: (batch_size, vocab_size), float tensor with counts or binary
        batch_size, vocab_size = x.size()

        # Reshape input to (batch_size, vocab_size, 1) so each element is a scalar "count"
        x = x.unsqueeze(-1)  # shape: (B, V, 1)

        # Apply φ to each element
        phi_output = self.phi(x)  # shape: (B, V, embed_dim)

        # Aggregate (mean over elements for permutation invariance)
        set_embedding = phi_output.mean(dim=1)  # shape: (B, embed_dim)

        # Apply ρ to get reconstruction
        out = self.rho(set_embedding)  # shape: (B, vocab_size)

        return out, set_embedding

    def training_step(self, batch, batch_idx):
        recon, _ = self(batch)
        loss = F.mse_loss(recon, batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        recon, _ = self(batch)
        loss = F.mse_loss(recon, batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def plot_clusters_matplotlib(embeddings, labels, title="t-SNE Clustering", figsize=(8, 6), random_seed: int = 42):
    """
    embeddings: numpy array of shape (n_samples, embedding_dim)
    labels: cluster labels (e.g., from KMeans or HDBSCAN), shape (n_samples,)
    """
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=random_seed)
    tsne_results = tsne.fit_transform(embeddings)

    # Prepare figure
    plt.figure(figsize=figsize)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 10:
        colors = plt.colormaps.get_cmap('tab10').resampled(len(unique_labels))
    elif len(unique_labels) < 20:
        colors = plt.colormaps.get_cmap('tab20').resampled(len(unique_labels))
    else:
        colors = plt.colormaps.get_cmap('viridis').resampled(len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=[colors(i)],
            label=f"Cluster {label}",
            s=60,
            alpha=0.8,
            edgecolors='w'
        )

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def extract_embeddings(model, dataloader):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            _, z = model(batch)
            embeddings.append(z)

    return torch.cat(embeddings, dim=0).cpu().numpy()


def main(filename: str, batch_size: int, max_epochs: int, random_seed: int) -> None:
    # Load the cards and the decklists
    cards = get_cards()
    remove = get_removables(cards)
    replace = get_replacements(cards, remove)

    decklists = load_and_update_decklists(filename)

    data = [
        (c, d['investigator_name'], d['tags'], d['xp'], {
            replace.get(i, i): n
            for i, n in d.get('slots', {}).items()
            if replace.get(i, i) not in remove
        })
        for c, d in decklists.items()
    ]

    # Extract the slots from the decklists
    idx, names, tags, xps, slots = zip(*data)
    vocab = sorted({e for d in slots for e in d.keys()})

    torch.manual_seed(random_seed)
    # Create the DataModule
    data_module = SetsDataModule(slots, vocab, batch_size=batch_size, vector_type="count")
    data_module.prepare_data()
    data_module.setup()

    # Get one batch
    for batch in data_module.train_dataloader():
        print(batch.shape)  # (batch_size, vocab_size)
        break

    # Initialize the Deep Sets model
    model = DeepSetsModel(vocab_size=len(vocab))

    # Train & validate the model
    trainer = Trainer(min_epochs=2, max_epochs=max_epochs, accelerator="auto")
    trainer.fit(model, data_module)

    # # Get embeddings for all sets (train + val)
    # # train_loader = data_module.train_dataloader()
    # # val_loader = data_module.val_dataloader()
    # all_loader = torch.utils.data.DataLoader(
    #     torch.utils.data.ConcatDataset([data_module.train_dataset, data_module.val_dataset]),
    #     batch_size=batch_size
    # )

    specific_data = [
        e
        for e in data
        if e[1] == 'Silas Marsh'
           and 'solo' in e[2]
           and (e[3] is None or e[3] == 0)]
    specific_idx, _, _, _, specific_slots = zip(*specific_data)

    specific_loader = torch.utils.data.DataLoader(
        SetsDataset(specific_slots, vocab, vector_type="count"),
        batch_size=batch_size
    )
    print('>>>', len(specific_data))

    embeddings = extract_embeddings(model, specific_loader)
    max_clusters = (len(embeddings) + 1) // 2

    best_k, best_score = 2, -1
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_seed)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, score

    # Cluster
    kmeans = KMeans(n_clusters=best_k, random_state=random_seed)
    best_labels = kmeans.fit_predict(embeddings)
    best_score = silhouette_score(embeddings, best_labels)

    print(f"Optimal clusters: {best_k} (Silhouette Score: {best_score:.3f})")
    if best_score > 0.7:
        print(' - [0.7, 1.0] Strong structure, clusters are well-separated and distinct.')
    elif best_score > 0.5:
        print(' - [0.5, 0.7] Reasonable structure, but clusters could be more distinct.')
    elif best_score > 0.25:
        print(' - [0.25, 0.5] Weak or overlapping structure, clusters are not well-defined.')
    elif best_score > 0.0:
        print(' - [< 0.25] No substantial structure; clustering may be arbitrary.')
    else:
        print(' - [<=0] Points are likely assigned to the wrong clusters.')
    print()

    title = f"Deck Clusters (k={best_k}, Silhouette={best_score:.2f})"
    plot_clusters_matplotlib(embeddings, best_labels, title=title, figsize=(8, 6), random_seed=random_seed)

    print('Done.')

    # # Apply KMeans on the 2D t-SNE embeddings
    # kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    # labels = kmeans.fit_predict(tsne_results)

    tsne = TSNE(n_components=2, perplexity=10, random_state=random_seed)
    tsne_results = tsne.fit_transform(embeddings)

    best_k, best_score = 2, -1
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_seed)
        labels = kmeans.fit_predict(tsne_results)
        score = silhouette_score(tsne_results, labels)
        if score > best_score:
            best_k, best_score = k, score

    # Cluster
    kmeans = KMeans(n_clusters=best_k, random_state=random_seed)
    best_labels = kmeans.fit_predict(tsne_results)
    best_score = silhouette_score(tsne_results, best_labels)

    print(f"Optimal clusters: {best_k} (Silhouette Score: {best_score:.3f})")
    if best_score > 0.7:
        print(' - [0.7, 1.0] Strong structure, clusters are well-separated and distinct.')
    elif best_score > 0.5:
        print(' - [0.5, 0.7] Reasonable structure, but clusters could be more distinct.')
    elif best_score > 0.25:
        print(' - [0.25, 0.5] Weak or overlapping structure, clusters are not well-defined.')
    elif best_score > 0.0:
        print(' - [< 0.25] No substantial structure; clustering may be arbitrary.')
    else:
        print(' - [<=0] Points are likely assigned to the wrong clusters.')
    print()

    results = {}
    for l, i, s in zip(best_labels, specific_idx, specific_slots):
        results.setdefault(int(l), {}).setdefault(i, s)

    with open('clusters.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main(
        filename=os.path.join(DATA_PATH, FILE_NAME),
        batch_size=8,
        max_epochs=5,  # 100,
        random_seed=1,
    )
