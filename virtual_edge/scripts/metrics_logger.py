import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
from collections.abc import Mapping

class CSVLogger:

    def __init__(self, path: Path, fieldnames: Iterable[str], mode: Literal["a", "w"] = "a"):
        if mode not in ("a", "w"):
            raise ValueError("mode must be either 'a' or 'w'")

        self.path = Path(path)
        self.fieldnames = tuple(fieldnames)
        self.mode = mode
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Fresh Run: recreate file and write header once
        if self.mode == "w":
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def write_row(self, row: Mapping[str, Any]) -> None:
        if row is None:
            raise ValueError("row cannot be None")

        write_header = (self.mode == "a" and (not self.path.exists() or self.path.stat().st_size == 0))
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in self.fieldnames})

class JSONLogger:

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Any) -> None:
        if is_dataclass(payload):
            payload = asdict(payload)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)