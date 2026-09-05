"""Load an SB3 progress.csv into plain numpy arrays, by column name."""
import csv

import numpy as np


def load_progress(csv_path: str) -> dict:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    columns = {}
    for key in rows[0].keys():
        values = []
        for row in rows:
            raw = row.get(key, "")
            values.append(float(raw) if raw not in ("", None) else np.nan)
        columns[key] = np.array(values, dtype=np.float64)
    return columns
