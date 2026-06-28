from __future__ import annotations


class PortfolioAnalyzerWindow:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The desktop window has been replaced by the Streamlit web app. "
            "Run `python -m portfolio_analyzer.main` or "
            "`streamlit run portfolio_analyzer/streamlit_app.py`."
        )
