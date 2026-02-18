from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class MetricsCSVLogger:
    def __init__(self, output_path: Path, fieldnames: Sequence[str] | None = None) -> None:
        self.output_path = output_path
        self._fieldnames = list(fieldnames) if fieldnames is not None else None

    @property
    def fieldnames(self) -> list[str] | None:
        if self._fieldnames is None:
            return None
        return list(self._fieldnames)

    def log(self, row: Mapping[str, Any]) -> None:
        if not row:
            raise ValueError("Metrics row must not be empty.")

        existing_fieldnames = self._existing_fieldnames()
        if self._fieldnames is None and existing_fieldnames is not None:
            self._fieldnames = existing_fieldnames
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

        missing = [name for name in self._fieldnames if name not in row]
        extras = [name for name in row if name not in self._fieldnames]
        if missing or extras:
            raise ValueError(
                "Metrics row keys must match logger fieldnames. "
                f"missing={missing}, extra={extras}, fieldnames={self._fieldnames}"
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        with self.output_path.open("a", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self._fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow({name: row[name] for name in self._fieldnames})

    def _existing_fieldnames(self) -> list[str] | None:
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return None

        with self.output_path.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file)
            header = next(reader, None)
        if header is None:
            return None
        return [str(name) for name in header]


def append_metrics_row(output_path: Path, row: Mapping[str, Any]) -> None:
    logger = MetricsCSVLogger(output_path)
    logger.log(row)
