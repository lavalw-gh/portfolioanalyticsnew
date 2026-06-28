from __future__ import annotations

from datetime import date
from typing import Callable

from .analysis_job import run_analysis_job
from .models import LoadedPortfolio


class AnalysisWorker:
    def __init__(
        self,
        loaded_portfolios: dict[str, LoadedPortfolio],
        selected_names: list[str],
        benchmark: str,
        start_requested: date,
        end_requested: date,
        preset: str,
        rf_annual: float,
        hampel_threshold: float,
        progress_callback: Callable[[str, int], None] | None = None,
    ):
        self.loaded_portfolios = loaded_portfolios
        self.selected_names = selected_names
        self.benchmark = benchmark
        self.start_requested = start_requested
        self.end_requested = end_requested
        self.preset = preset
        self.rf_annual = rf_annual
        self.hampel_threshold = hampel_threshold
        self.progress_callback = progress_callback

    def run(self) -> dict:
        return run_analysis_job(
            self.loaded_portfolios,
            self.selected_names,
            self.benchmark,
            self.start_requested,
            self.end_requested,
            self.preset,
            self.rf_annual,
            self.hampel_threshold,
            progress_callback=self.progress_callback,
        )
