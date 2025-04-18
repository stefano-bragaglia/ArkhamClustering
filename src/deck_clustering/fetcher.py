""" Fetch deck
"""
import json
import os
from datetime import date, timedelta
from datetime import datetime
from typing import Any, Generator

import requests
from tqdm import tqdm

API_URL_DECKLIST = "https://arkhamdb.com/api/public/decklist/{decklist_id}"
API_URL_DECKLISTS_BY_DATE = "https://arkhamdb.com/api/public/decklists/by_date/{date}"

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
FILE_NAME = 'decklist.json'
DATE_FMT = "%Y-%m-%d"


def load_and_update_decklists(filename) -> dict[str, Any]:
    """ Load and update decklists from the ArkhamDB API.

    This function checks if the specified file exists. If it does, it loads the decklists from the file.
    If it doesn't, it initializes an empty dictionary.
    It then fetches the most recent date from the loaded decklists and the current date.
    It iterates through the dates in that range and fetches decklists for each date.
    If a decklist is successfully fetched, it is prepared and added to the dictionary.
    Finally, the updated decklists are saved back to the file.

    :param filename: The path to the file where decklists are stored.
    :return: A dictionary containing the updated decklists.
    """
    # Check if the file exists and load the decklists
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            decklists = json.load(file)
    else:
        decklists = dict()

    # Get the most recent date from the loaded decklists
    start_date = get_most_recent_date(decklists)
    end_date = get_today_date()
    dates = daterange(start_date, end_date)
    days = get_number_of_days(start_date, end_date)

    # Fetch decklists for each date in the range
    for single_date in tqdm(dates, desc='Fetching decklists', total=days, ncols=120):
        for decklist in get_decklists_by_date(single_date):
            if decklist != 'error':
                ident = decklist["id"]
                decklists[ident] = prepare_decklist(decklist)

    # Save the updated decklists to the file
    with open(filename, 'w') as file:
        json.dump(decklists, file, indent=2)

    return decklists


def get_most_recent_date(decklists: dict[str, Any]) -> date:
    """ Get the most recent date for the decklists.

    This function checks if the decklists is empty. If it is, it fetches the initial date from the ArkhamDB API.
    If the decklists is not empty, it retrieves the latest creation date among the decklists.

    :param decklists: A dictionary containing decklists, where the keys are decklist IDs and the values are dictionaries representing the decklists.
    :return: A date object representing the most recent date for the decklists.
    """
    if not decklists:
        return get_initial_date()

    return max(get_creation_date(deck) for deck in decklists.values())


def get_initial_date() -> date:
    """ Fetch the initial date from the ArkhamDB API.

    This function retrieves the creation date of a decklist with ID 1 from the ArkhamDB API.
    The date is extracted from the response and returned as a date object.
    :return: A date object representing the creation date of the decklist.
    """
    url = API_URL_DECKLIST.format(decklist_id=1)
    with requests.get(url=url) as r:
        deck = r.json()

    return get_creation_date(deck)


def get_creation_date(deck: dict[str, Any]) -> date:
    """ Extract the creation date from a deck dictionary.

    The date is extracted from the 'date_creation' key and converted to a date object.

    :param deck: A dictionary representing a deck, which must contain the 'date_creation' key.
    :return: A date object representing the creation date of the deck.
    """
    content = deck['date_creation'].split('T')[0]
    result = datetime.strptime(content, DATE_FMT).date()

    return result


def get_today_date() -> date:
    """ Get today's date.

    This function retrieves the current date using the datetime module.

    :return: A date object representing today's date.
    """
    return datetime.today().date()


def daterange(start_date: date, end_date: date) -> Generator[date, None, None]:
    """ Generate a range of dates between two dates.

    This function takes two date objects as input and yields each date in the range from start_date to end_date (exclusive).
    :param start_date: The start date.
    :param end_date: The end date.
    :return: A generator that yields date objects in the range.
    """
    days = get_number_of_days(start_date, end_date)
    for n in range(days):
        yield start_date + timedelta(n)


def get_number_of_days(start_date: date, end_date: date) -> int:
    """ Calculate the number of days between two dates.

    This function takes two date objects as input and returns the number of days between them as an integer.

    :param start_date: The start date.
    :param end_date: The end date.
    :return: An integer representing the number of days between the two dates.
    """
    time_delta = end_date - start_date
    result = int(time_delta.days)

    return result


def get_decklists_by_date(specific_date: date) -> list[dict[str, Any]]:
    """ Fetch decklists from the ArkhamDB API for a specific date.

    This function takes a date object as input and retrieves the decklists for that date from the API.
    The date is formatted as a string in the format YYYY-MM-DD.

    :param specific_date: The date for which to fetch decklists.
    :return: A list of dictionaries representing the decklists for the specified date.
    """
    url = API_URL_DECKLISTS_BY_DATE.format(date=specific_date.strftime(DATE_FMT))
    with requests.get(url=url) as r:
        return r.json()


def prepare_decklist(decklist: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare a decklist by cleaning it up.

    This function takes a decklist dictionary as input and processes it to ensure that:
    * the 'tags' field is a list of strings (beginner, solo, multiplayer, theme).

    :param decklist: A dictionary representing a decklist.
    :return: The processed decklist dictionary.
    """
    decklist['tags'] = [t.strip() for t in decklist.get('tags', '').split(',') if t.strip()]

    return decklist


def main(filename: str) -> None:
    decklists = load_and_update_decklists(filename)

    print(json.dumps(decklists, indent=2))
    print('Done.')


if __name__ == '__main__':
    main(
        filename=os.path.join(DATA_PATH, FILE_NAME),
    )
