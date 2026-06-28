from __future__ import annotations


class DataFrameModel:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Desktop table models are no longer used by the Streamlit app.")


class MplCanvas:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Desktop chart canvases are no longer used by the Streamlit app.")
