import json

from transfer import default_params
from commands import apply_command
from evaluate import jsonl_append


def test_jsonl_append_writes_one_line(tmp_path):
    path = tmp_path / "log.jsonl"
    jsonl_append(str(path), {"a": 1})
    jsonl_append(str(path), {"b": 2})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
