"""Load an SB3 progress.csv into plain numpy arrays, by column name."""
import csv

import numpy as np


def load_progress(csv_path: str) -> dict:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    columns = {}
    for key in fieldnames:
        values = []
        for row in rows:
            raw = row.get(key, "")
            if raw in ("", None):
                values.append(np.nan)
            else:
                v = float(raw)
                values.append(np.nan if np.isinf(v) else v)
        columns[key] = np.array(values, dtype=np.float64)
    return columns
