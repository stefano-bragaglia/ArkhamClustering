""" Rendering.
"""
import json
from collections import Counter

import numpy as np

from fetcher import get_cards

SUBTYPES = {
    '': 0, 'Basic Weakness': 1, 'Weakness': 2,
}
TYPES = {
    '': 0,
    'Asset': 1,
    'Event': 2,
    'Skill': 4,
    'Treachery': 8,
    'Enemy': 16,
    'Story': 32,
    'Location': 64,
    'Investigator': 128,
}
PERMANENT = {
    None: 0,
    False: 0,
    True: 1,
}
SLOTS = {
    'Hand': 1,
    'Hand x2': 2,
    'Arcane': 4,
    'Arcane x2': 8,
    'Accessory': 16,
    'Body': 32,
    'Tarot': 64,
    'Ally': 128,
    '': 256,
}



def rendering(
        decks: list[dict[str, int]],
        labels: np.ndarray[tuple[int, ...], np.dtype],
        names: dict[str, tuple[int, int, int, int, str]],
) -> None:
    for num, length in Counter(labels).most_common():
        if num == -1 or length <= 1:
            continue

        print(f'Cluster #{num} ({length} deck/s)')
        cluster_decks = [
            deck
            for deck, label in zip(decks, labels)
            if label == num
        ]
        print(cluster_decks)

        cards = {
            code: []
            for code in sorted({
                card
                for deck in cluster_decks
                for card in deck.keys()
            }, key=lambda x: names[x])
        }
        print(cards)

        for deck in cluster_decks:
            for code in cards.keys():
                num = deck.get(code, 0)
                cards[code].append('         ' if num == 0 else f' {code} {num} ')

        for code, line in cards.items():
            ubiquity = 100 * sum(1 for l in line if l != '         ') / len(line)
            intensity = max(1, int(256 * (100 - ubiquity) / 100))
            content = ''.join([*line, f'{ubiquity:5.1f}%', ' ', names[code][-1]])
            print(f'\033[38;2;{intensity};{intensity};{intensity}m{content}\033[0m')

        print()


def main() -> None:
    cards = {c['code']: c for c in get_cards()}
    combine = lambda x: sum(SLOTS[p.strip()] for p in x.split('.'))
    names = {
        **{
            c['code']: (
                SUBTYPES[c.get('subtype_name', '')],
                TYPES[c.get('type_name', '')],
                PERMANENT[c.get('permanent', False)],
                combine(c.get('slot', '')),
                c.get('name', '') + ''.join('•' * c.get('xp', 0))
            ) for c in cards.values()
        },
        '01117': (SUBTYPES[''], TYPES['Asset'], PERMANENT[False], combine('Ally'), 'Lita Chantler'),
        '02040': (SUBTYPES[''], TYPES['Asset'], PERMANENT[False], combine('Ally'), 'Dr. Henry Armitage'),
        '02080': (SUBTYPES[''], TYPES['Asset'], PERMANENT[False], combine('Ally'), 'Dr. Francis Morgan'),
        '07083': (SUBTYPES[''], TYPES['Asset'], PERMANENT[False], combine('Ally'), 'Elina Harper'),
        '07179': (SUBTYPES[''], TYPES['Asset'], PERMANENT[False], combine(''), 'Waveworn Idol'),
        '07180': (SUBTYPES[''], TYPES['Asset'], PERMANENT[False], combine(''), 'Awakened Mantle'),
        '07181': (SUBTYPES[''], TYPES['Asset'], PERMANENT[False], combine(''), 'Headdress of Y\'ha-nthlei'),
    }

    with open('clusters.json', 'r') as f:
        clusters = json.load(f)

    for num, decks in clusters.items():
        length = len(decks)
        content = f'Cluster #{num} ({length} deck/s)'
        print(f'\033[38;2;{255};{0};{0}m{content}\033[0m')
        content = ''.join(f'   {code:05} ' for code in decks.keys())
        print(f'\033[38;2;{0};{0};{255}m{content}\033[0m')
        codes = {c: names[c] for d in decks.values() for c in d.keys()}
        max_length = 0
        for code, (_, _, _, _, name) in sorted(codes.items(), key=lambda x: x[1]):
            line, hits = [], 0
            for deck in decks.values():
                if code in deck:
                    line.append(f' {code} {deck[code]} ')
                    hits += 1
                else:
                    line.append('         ')
            # if hits == 0:
            #     continue
            ubiquity = 100 * hits / length
            content = ''.join([*line, f'{ubiquity:5.1f}% {name}'])
            max_length = max(max_length, len(content))
            shade = max(1, int(256 * (100 - ubiquity) / 100))
            print(f'\033[38;2;{shade};{shade};{shade}m{content}\033[0m')

        print('-' * max_length)
        print()


if __name__ == '__main__':
    main()
