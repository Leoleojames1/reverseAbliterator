from typing import List, Tuple
from itertools import islice
from sklearn.model_selection import train_test_split

def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

def prepare_dataset(dataset: Tuple[List[str], List[str]]|List[str]) -> Tuple[List[str], List[str]]:
    if len(dataset) != 2:
        train, test = train_test_split(dataset, test_size=0.1, random_state=42)
    else:
        train, test = dataset
    return train, test