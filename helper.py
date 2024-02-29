from typing import List, Dict
from collections import defaultdict
import csv

def read_csv(file_path:str) -> List[dict]:
    """
    Reads the given path and returns a list of dicts.
    Dicts' keys are the header's names.

    Args:
        file_path (str)

    Returns:
        _type_: _description_
    """
    data = []
    with open(file_path, newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def generate_transposed_dict(csv_data:List[dict]) -> Dict[list]:
    """Creates a Dict of lists from a List of Dicts for analytical purposes.

    Args:
        csv_data (List[dict])

    Returns:
        Dict[list]
    """

    headers = list(csv_data[0].keys())

    transposed = defaultdict(list)

    for attack_record in csv_data:
        for key in headers:
            transposed[key].append(attack_record[key])

    return transposed
