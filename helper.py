from typing import List
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
