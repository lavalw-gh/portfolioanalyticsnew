from __future__ import annotations

from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st


if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    from portfolio_analyzer.analysis_job import run_analysis_job
    from portfolio_analyzer.charts import make_portfolio_altair_chart
    from portfolio_analyzer.constants import (
        APP_TITLE,
        BENCHMARK_OPTIONS,
        DATE_PRESETS,
        DEFAULT_BENCHMARK,
    )
    from portfolio_analyzer.data_loading import load_transactions_from_bytes
    from portfolio_analyzer.date_utils import compute_preset_dates
    from portfolio_analyzer.docx_export import write_docx
    from portfolio_analyzer.models import LoadedPortfolio
else:
    from .analysis_job import run_analysis_job
    from .charts import make_portfolio_altair_chart
    from .constants import (
        APP_TITLE,
        BENCHMARK_OPTIONS,
        DATE_PRESETS,
        DEFAULT_BENCHMARK,
    )
    from .data_loading import load_transactions_from_bytes
    from .date_utils import compute_preset_dates
    from .docx_export import write_docx
    from .models import LoadedPortfolio


def _portfolio_name(filename: str, existing: set[str]) -> str:
    pname = Path(filename).stem
    base_name = pname
    counter = 2
    while pname in existing:
        pname = f"{base_name} ({counter})"
        counter += 1
    return pname


@st.cache_data(show_spinner=False)
def _load_uploaded_portfolios(
    file_payloads: tuple[tuple[str, bytes], ...],
) -> tuple[dict[str, LoadedPortfolio], list[str]]:
    loaded: dict[str, LoadedPortfolio] = {}
    errors: list[str] = []
    for filename, data in file_payloads:
        try:
            tx, conversion_factors, yahoo_names, notes = load_transactions_from_bytes(data)
            pname = _portfolio_name(filename, set(loaded))
            loaded[pname] = LoadedPortfolio(
                name=pname,
                tx=tx,
                conversion_factors=conversion_factors,
                yahoo_names=yahoo_names,
                normalization_notes=notes,
                source_path=filename,
            )
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    return loaded, errors


def _payload_signature(file_payloads: tuple[tuple[str, bytes], ...]) -> tuple[tuple[str, str], ...]:
    return tuple((name, sha256(data).hexdigest()) for name, data in file_payloads)


def _selectbox_allow_custom(label: str, options: list[str], default: str) -> str:
    index = options.index(default) if default in options else 0
    try:
        return st.selectbox(
            label,
            options,
            index=index,
            accept_new_options=True,
        )
    except TypeError:
        return st.text_input(label, value=default)


def _analysis_key(
    loaded_portfolios: dict[str, LoadedPortfolio],
    file_payloads: tuple[tuple[str, bytes], ...],
    benchmark: str,
    start_requested: date,
    end_requested: date,
    preset: str,
    rf_annual: float,
    hampel_threshold: float,
) -> tuple:
    return (
        _payload_signature(file_payloads),
        tuple(loaded_portfolios.keys()),
        benchmark,
        start_requested.isoformat(),
        end_requested.isoformat(),
        preset,
        round(float(rf_annual), 8),
        round(float(hampel_threshold), 6),
    )


def _run_analysis(
    loaded_portfolios: dict[str, LoadedPortfolio],
    benchmark: str,
    start_requested: date,
    end_requested: date,
    preset: str,
    rf_annual: float,
    hampel_threshold: float,
) -> dict:
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def progress(message: str, pct: int) -> None:
        progress_text.text(message)
        progress_bar.progress(int(max(0, min(100, pct))))

    results = run_analysis_job(
        loaded_portfolios,
        list(loaded_portfolios.keys()),
        benchmark,
        start_requested,
        end_requested,
        preset,
        rf_annual,
        hampel_threshold,
        progress_callback=progress,
    )
    progress_text.text("Calculations complete.")
    progress_bar.progress(100)
    return results


def _build_docx_bytes(results: dict) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "portfolio_analysis.docx"
        write_docx(results, str(output_path))
        return output_path.read_bytes()


def _default_docx_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"portfolio_analysis_{stamp}.docx"


def _show_portfolio(results: dict, portfolio_name: str) -> None:
    pdata = results.get("portfolio_data", {}).get(portfolio_name)
    if pdata is None:
        st.warning("The selected portfolio is not available in the latest results.")
        return

    st.subheader(f"Portfolio: {portfolio_name}")
    chart = make_portfolio_altair_chart(
        portfolio_name,
        pdata,
        bench_cum_window=results.get("bench_cum_window", pd.Series(dtype=float)),
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Key metrics")
    st.dataframe(
        pdata.get("metrics_df", pd.DataFrame()),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Holdings")
    st.dataframe(
        pdata.get("holdings_df", pd.DataFrame()),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Price data quality")
    st.dataframe(
        results.get("price_quality_df", pd.DataFrame()),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Inputs")
        uploaded_files = st.file_uploader(
            "Upload CSV file(s)",
            type=["csv"],
            accept_multiple_files=True,
        )
        file_payloads = tuple(
            (uploaded_file.name, uploaded_file.getvalue())
            for uploaded_file in uploaded_files
        )

        loaded_portfolios: dict[str, LoadedPortfolio] = {}
        load_errors: list[str] = []
        if file_payloads:
            with st.spinner("Loading CSV files..."):
                loaded_portfolios, load_errors = _load_uploaded_portfolios(file_payloads)
            st.caption(
                "\n".join(
                    f"- {portfolio.name}: {portfolio.source_path}"
                    for portfolio in loaded_portfolios.values()
                )
                or "No files loaded"
            )
        else:
            st.caption("No files loaded")

        if load_errors:
            st.error("\n".join(load_errors))

        st.header("Selection")
        benchmark = _selectbox_allow_custom(
            "Benchmark index",
            BENCHMARK_OPTIONS,
            DEFAULT_BENCHMARK,
        ).strip() or DEFAULT_BENCHMARK

        portfolio_names = list(loaded_portfolios.keys())
        selected_portfolio = None
        if portfolio_names:
            selected_portfolio = st.selectbox(
                "Portfolio to display",
                portfolio_names,
                index=0,
            )

        st.header("Date range")
        preset = st.selectbox(
            "Preset",
            DATE_PRESETS,
            index=DATE_PRESETS.index("3 months"),
        )
        preset_start, preset_end = compute_preset_dates(preset, date.today())
        if preset == "Custom":
            start_requested = st.date_input("Start", value=preset_start)
            end_requested = st.date_input("End", value=preset_end)
        else:
            st.date_input("Start", value=preset_start, disabled=True)
            st.date_input("End", value=preset_end, disabled=True)
            start_requested = preset_start
            end_requested = preset_end

        st.header("Risk-free rate")
        rf_pct = st.number_input(
            "Annual (%)",
            min_value=-10.0,
            max_value=50.0,
            value=2.0,
            step=0.25,
            format="%.2f",
        )

        st.header("Daily move filter")
        hampel_threshold = st.number_input(
            "Max daily move (%)",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            format="%.1f",
        )

        st.header("Actions")
        run_clicked = st.button(
            "Run / Refresh",
            disabled=not bool(loaded_portfolios),
            use_container_width=True,
        )

    if not loaded_portfolios:
        st.info("Upload at least one CSV file to begin.")
        return

    if start_requested > end_requested:
        st.error("Start date must be on or before end date.")
        return

    rf_annual = float(rf_pct) / 100.0
    current_key = _analysis_key(
        loaded_portfolios,
        file_payloads,
        benchmark,
        start_requested,
        end_requested,
        preset,
        rf_annual,
        hampel_threshold,
    )

    if run_clicked:
        try:
            st.session_state["latest_results"] = _run_analysis(
                loaded_portfolios,
                benchmark,
                start_requested,
                end_requested,
                preset,
                rf_annual,
                hampel_threshold,
            )
            st.session_state["latest_key"] = current_key
            st.session_state.pop("docx_bytes", None)
            st.session_state.pop("docx_name", None)
        except Exception as exc:
            st.session_state.pop("latest_results", None)
            st.session_state.pop("latest_key", None)
            st.error(f"Analysis error: {exc}")

    results = st.session_state.get("latest_results")
    if not results:
        st.info("Press Run / Refresh to calculate the loaded portfolio files.")
        return

    if st.session_state.get("latest_key") != current_key:
        st.warning("Inputs have changed since the latest calculation. Press Run / Refresh before exporting.")

    display_names = list(results.get("selected_names", []))
    if selected_portfolio not in display_names and display_names:
        selected_portfolio = display_names[0]

    cols = st.columns([1, 1])
    with cols[0]:
        st.metric("Benchmark", results.get("benchmark", ""))
        st.metric(
            "Date range",
            f"{results.get('start_effective')} to {results.get('end_effective')}",
        )
    with cols[1]:
        export_clicked = st.button("Export to Word", use_container_width=True)
        if export_clicked:
            try:
                st.session_state["docx_bytes"] = _build_docx_bytes(results)
                st.session_state["docx_name"] = _default_docx_name()
            except Exception as exc:
                st.error(f"Export failed: {exc}")
        if st.session_state.get("docx_bytes"):
            st.download_button(
                "Download Word document",
                data=st.session_state["docx_bytes"],
                file_name=st.session_state.get("docx_name", _default_docx_name()),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

    if selected_portfolio:
        _show_portfolio(results, selected_portfolio)


if __name__ == "__main__":
    main()
