from __future__ import annotations

from pathlib import Path

import pandas as pd

VALUE_BLUE = "#2563EB"
RETURN_GREEN = "#059669"
BENCHMARK_AMBER = "#D97706"


def _chart_dataframe(
    pdata: dict,
    bench_cum_window: pd.Series | None = None,
) -> pd.DataFrame:
    value_series = pdata.get("portfolio_value_window", pd.Series(dtype=float)).dropna()
    cum_series = pdata.get("twr_cum", pd.Series(dtype=float)).dropna()
    bench_series = (
        bench_cum_window.dropna()
        if bench_cum_window is not None
        else pd.Series(dtype=float)
    )

    indexes = []
    for series in (value_series, cum_series, bench_series):
        if not series.empty:
            indexes.append(series.index)
    if not indexes:
        return pd.DataFrame(
            columns=[
                "Date",
                "Portfolio Value",
                "Portfolio Return",
                "Benchmark Return",
            ]
        )

    index = indexes[0]
    for extra_index in indexes[1:]:
        index = index.union(extra_index)
    index = pd.DatetimeIndex(index).sort_values()

    df = pd.DataFrame({"Date": index})
    df["Portfolio Value"] = value_series.reindex(index).astype(float).values
    df["Portfolio Return"] = (cum_series.reindex(index).astype(float) * 100.0).values
    df["Benchmark Return"] = (bench_series.reindex(index).astype(float) * 100.0).values
    return df


def _require_altair():
    try:
        import altair as alt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Altair is required for charts. Install dependencies with "
            "`pip install streamlit altair vl-convert-python`."
        ) from exc
    return alt


def make_portfolio_altair_chart(
    pname: str,
    pdata: dict,
    bench_cum_window: pd.Series | None = None,
    *,
    width: int | str = "container",
    height: int = 420,
):
    alt = _require_altair()
    df = _chart_dataframe(pdata, bench_cum_window)

    if df.empty:
        return (
            alt.Chart(pd.DataFrame({"Message": ["No chart data available"]}))
            .mark_text(size=16, color="#475569")
            .encode(text="Message:N")
            .properties(title=pname, width=width, height=height)
        )

    base = alt.Chart(df).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%d %b %Y")),
    )

    value_axis = alt.Axis(
        title="Portfolio Value (GBP)",
        orient="left",
        format=",.0f",
        titleColor=VALUE_BLUE,
        labelColor=VALUE_BLUE,
    )
    value_area = base.mark_area(color=VALUE_BLUE, opacity=0.20).encode(
        y=alt.Y("Portfolio Value:Q", axis=value_axis)
    )
    value_line = base.mark_line(color=VALUE_BLUE, opacity=0.70, strokeWidth=1.5).encode(
        y=alt.Y("Portfolio Value:Q", axis=value_axis)
    )

    returns_df = df.melt(
        id_vars=["Date"],
        value_vars=["Portfolio Return", "Benchmark Return"],
        var_name="Series",
        value_name="Cumulative Return",
    ).dropna(subset=["Cumulative Return"])

    returns = (
        alt.Chart(returns_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(format="%d %b %Y")),
            y=alt.Y(
                "Cumulative Return:Q",
                axis=alt.Axis(
                    title="Cumulative Return (%)",
                    orient="right",
                    format=".1f",
                    titleColor=RETURN_GREEN,
                    labelColor=RETURN_GREEN,
                ),
            ),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Portfolio Return", "Benchmark Return"],
                    range=[RETURN_GREEN, BENCHMARK_AMBER],
                ),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            strokeDash=alt.StrokeDash(
                "Series:N",
                scale=alt.Scale(
                    domain=["Portfolio Return", "Benchmark Return"],
                    range=[[1, 0], [6, 4]],
                ),
                legend=None,
            ),
        )
    )

    return (
        alt.layer(value_area, value_line, returns)
        .resolve_scale(y="independent")
        .properties(title=pname, width=width, height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True)
    )


def make_chart_image(
    pname: str,
    pdata: dict,
    outputdir: Path,
    bench_cum_window: pd.Series | None = None,
) -> Path:
    outputdir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in pname)
    chartpath = outputdir / f"{safe_name}.png"

    try:
        import vl_convert as vlc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "vl-convert-python is required to export Altair charts to Word. "
            "Install it with `pip install vl-convert-python`."
        ) from exc

    chart = make_portfolio_altair_chart(
        pname,
        pdata,
        bench_cum_window=bench_cum_window,
        width=1100,
        height=450,
    )
    png_bytes = vlc.vegalite_to_png(chart.to_json())
    chartpath.write_bytes(png_bytes)
    return chartpath
