from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LoadedPortfolio:
    name: str
    tx: pd.DataFrame
    conversion_factors: dict
    yahoo_names: dict
    normalization_notes: list[str]
    source_path: str
