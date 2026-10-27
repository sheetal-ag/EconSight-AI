"""Plotly figure builders: data in, figure out, no Streamlit code.

Design standards used throughout:
- Titles state the finding; the subtitle gives units, period and caveats.
- Colorblind-safe palette (Okabe-Ito), max 6 lines, direct labels instead of legends.
- Source line on every chart; bar-like charts start at zero; truncated axes are disclosed.
- No dual-axis charts (their scales are arbitrary and mislead). Compare with indexed
  change, small multiples, heatmaps or scatter plots instead.
- Neutral greys with transparency so charts read well in light and dark themes.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tools import BASELINE, RECOVERY_TOLERANCE, STATE_ABBR, label

SOURCE = "Source: U.S. Bureau of Labor Statistics via FRED. Seasonally adjusted, monthly."
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]  # Okabe-Ito
MUTED = "#8A8F98"
GRID = "rgba(128,128,128,0.22)"
FONT = "Inter, 'Segoe UI', Arial, sans-serif"
LABEL = {"unemployment": "Unemployment rate", "participation": "Labor force participation rate"}


def _style(fig, title, subtitle, height=480, right_margin=24):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{MUTED}'>{subtitle}</span>",
                   x=0, xanchor="left", y=0.96, yanchor="top"),
        font=dict(family=FONT, size=13), height=height,
        margin=dict(l=10, r=right_margin, t=100, b=92),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=13), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.add_annotation(text=SOURCE, xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="top",
                       yshift=-62, showarrow=False, font=dict(size=11, color=MUTED))
    return fig


def trend_chart(full, states, metric, mode="Level", start=None, end=None):
    """Line chart for up to 6 states with direct labels and the recession band."""
    data = full[states]
    indexed = mode != "Level"
    if indexed:
        data = data - full.loc[BASELINE, states]
    data = data.loc[start:end]
    if indexed:
        unit, ytitle = "points", f"Change since {label(BASELINE)} (percentage points)"
    else:
        unit, ytitle = "%", f"{LABEL[metric]} (%)"

    if metric == "unemployment":
        s = data.max().idxmax()
        val = f"+{data[s].max():.1f} points" if indexed else f"{data[s].max():.1f}%"
        headline = f"{s} had the highest unemployment peak of the selected states ({val} in {label(data[s].idxmax())})"
    else:
        chg = data - full.loc[BASELINE, states] if not indexed else data
        s = chg.min().idxmin()
        headline = f"{s} saw the deepest participation drop of the selected states ({chg[s].min():+.1f} points vs {label(BASELINE)})"
    note = "" if (indexed or metric == "unemployment") else " Y-axis does not start at zero."
    sub = f"{LABEL[metric]}, {label(data.index.min())} to {label(data.index.max())}.{note}"

    fig = go.Figure()
    gap = 0.06 * float(data.max().max() - data.min().min() or 1)       # min spacing between end labels
    label_y, prev = {}, None
    for st_ in sorted(states, key=lambda n: data[n].iloc[-1]):
        y = float(data[st_].iloc[-1])
        label_y[st_] = y if prev is None else max(y, prev + gap)
        prev = label_y[st_]
    for i, st_ in enumerate(states):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=data.index, y=data[st_], name=st_, mode="lines", line=dict(width=2.6, color=color),
                                 hovertemplate=f"{st_}: %{{y:.1f}} {unit}<extra></extra>"))
        fig.add_annotation(x=data.index[-1], y=label_y[st_], text=f"<b>{st_}</b>", xanchor="left", xshift=6,
                           showarrow=False, font=dict(color=color, size=12))
    if data.index.min() <= pd.Timestamp("2020-04-01") and data.index.max() >= pd.Timestamp("2020-02-01"):
        fig.add_vrect(x0="2020-02-01", x1="2020-04-01", fillcolor="rgba(128,128,128,0.16)", line_width=0,
                      layer="below", annotation_text="US recession (NBER)", annotation_position="top left",
                      annotation_font=dict(size=11, color=MUTED))
    if indexed:
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
    if metric == "unemployment" and not indexed:
        fig.update_yaxes(rangemode="tozero")
    fig.update_yaxes(title_text=ytitle)
    fig.update_layout(hovermode="x unified")
    return _style(fig, headline, sub, right_margin=110)


def impact_dumbbell(summary):
    """Before vs peak unemployment for every state, sorted by size of the jump."""
    s = summary.sort_values("unemployment_increase")
    order = list(s.index)
    xs, ys = [], []
    for name, r in s.iterrows():
        xs += [r["baseline_unemployment"], r["peak_unemployment"], None]
        ys += [name, name, None]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=GRID, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s["baseline_unemployment"], y=order, mode="markers", name=f"{label(BASELINE)} (before)",
                             marker=dict(size=9, color=MUTED), hovertemplate="%{y}: %{x:.1f}% before<extra></extra>"))
    fig.add_trace(go.Scatter(x=s["peak_unemployment"], y=order, mode="markers", name="Peak", customdata=s["peak_month"],
                             marker=dict(size=10, color=PALETTE[1]),
                             hovertemplate="%{y}: %{x:.1f}% at peak (%{customdata})<extra></extra>"))
    top = order[-1]
    fig.update_yaxes(categoryorder="array", categoryarray=order, tickfont=dict(size=11), gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Unemployment rate (%)", rangemode="tozero", showgrid=True, gridcolor=GRID)
    _style(fig, f"{top} saw the largest jump in unemployment: {s.loc[top, 'baseline_unemployment']:.1f}% to "
                f"{s.loc[top, 'peak_unemployment']:.1f}%",
           f"Each state before COVID ({label(BASELINE)}) versus its peak. Sorted by size of the increase.",
           height=max(520, 20 * len(s) + 190))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.04, x=0, yanchor="bottom"))
    return fig


def recovery_bars(summary):
    rec = summary.dropna(subset=["months_to_recover"]).sort_values("months_to_recover")
    missing = list(summary.index[summary["months_to_recover"].isna()])
    sub = (f"Months from the unemployment peak until the rate was within {RECOVERY_TOLERANCE:g} point of its "
           f"{label(BASELINE)} level.")
    if missing:
        sub += f" Not yet recovered: {', '.join(missing[:8])}{' and more' if len(missing) > 8 else ''}."
    fig = go.Figure()
    if rec.empty:
        title = "No state had recovered by the end of the period"
    else:
        fig.add_trace(go.Bar(x=rec["months_to_recover"], y=rec.index, orientation="h", marker_color=PALETTE[0],
                             hovertemplate="%{y}: %{x:.0f} months<extra></extra>"))
        title = (f"{len(rec)} of {len(summary)} states recovered; fastest was {rec.index[0]} "
                 f"({int(rec['months_to_recover'].iloc[0])} months)")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Months to recover", showgrid=True, gridcolor=GRID, rangemode="tozero")
    return _style(fig, title, sub, height=max(520, 20 * max(len(rec), 5) + 190))


def heatmap(full, metric, start=None, end=None):
    """States x months. Unemployment as level; participation as change vs baseline (diverging scale)."""
    if metric == "unemployment":
        z, scale, zmid, cbar = full, "YlOrRd", None, "%"
        order = z.loc[start:end].max().sort_values(ascending=False).index
        sub = "Unemployment rate (%) by state and month. States sorted by their peak, highest first."
        top = order[0]
        title = f"{top} was hit hardest: unemployment peaked at {z.loc[start:end, top].max():.1f}% in {label(z.loc[start:end, top].idxmax())}"
    else:
        z, scale, zmid, cbar = full - full.loc[BASELINE], "RdBu", 0, "points"
        order = z.loc[start:end].min().sort_values().index
        sub = f"Change in participation rate vs {label(BASELINE)} (percentage points). Red = below baseline."
        sel = z.loc[start:end]
        title = (f"Participation dipped below its {label(BASELINE)} level in {int((sel.min() < 0).sum())} of {sel.shape[1]} states; "
                 f"deepest in {order[0]} ({sel[order[0]].min():+.1f} points)")
    z = z.loc[start:end, order].T
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale=scale, zmid=zmid,
                               colorbar=dict(title=cbar, thickness=12),
                               hovertemplate="%{y}, %{x|%b %Y}: %{z:.1f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", dtick=1, tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)")
    return _style(fig, title, sub, height=max(560, 17 * len(z) + 190), right_margin=40)


def us_map(full, metric, month):
    vals = full.loc[month]
    plot = pd.DataFrame({"state": vals.index, "abbr": [STATE_ABBR[s] for s in vals.index], "value": vals.values})
    scale = "YlOrRd" if metric == "unemployment" else "Blues"
    fig = px.choropleth(plot, locations="abbr", locationmode="USA-states", color="value", scope="usa",
                        color_continuous_scale=scale, range_color=(float(full.min().min()), float(full.max().max())),
                        hover_name="state")
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>%{z:.1f}%<extra></extra>", marker_line_color="white")
    fig.update_layout(coloraxis_colorbar=dict(title="%", thickness=12), geo=dict(bgcolor="rgba(0,0,0,0)"))
    top = vals.idxmax() if metric == "unemployment" else vals.idxmin()
    word = "highest unemployment" if metric == "unemployment" else "lowest participation"
    _style(fig, f"{top} had the {word} in {label(month)}: {vals[top]:.1f}%",
           f"{LABEL[metric]} by state.",
           height=520, right_margin=10)
    return fig


def relationship_scatter(frame, label_n=6):
    x, y = frame["unemployment_increase"], frame["participation_change"]
    slope, intercept = np.polyfit(x, y, 1)
    r = float(x.corr(y))
    if r <= -0.3:
        title = f"States with bigger unemployment spikes also saw bigger participation drops (r = {r:.2f})"
    elif r >= 0.3:
        title = f"States with bigger unemployment spikes saw smaller participation drops (r = {r:.2f})"
    else:
        title = f"No clear link between unemployment spikes and participation drops across states (r = {r:.2f})"
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", text=frame.index, name="State",
                               marker=dict(size=10, color=PALETTE[0], opacity=0.85, line=dict(width=1, color="white")),
                               hovertemplate="<b>%{text}</b><br>Unemployment +%{x:.1f} pts<br>Participation %{y:+.1f} pts<extra></extra>"))
    xs = np.array([x.min(), x.max()])
    fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, mode="lines", hoverinfo="skip",
                             line=dict(color=PALETTE[1], width=2, dash="dash")))
    resid = (y - (intercept + slope * x)).abs().sort_values(ascending=False).head(label_n).index
    for s in resid:
        fig.add_annotation(x=x[s], y=y[s], text=s, showarrow=False, yshift=12, font=dict(size=11, color=MUTED))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(title_text="Rise in unemployment rate, Feb 2020 to peak (points)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text="Change in participation, Feb 2020 to lowest (points)")
    return _style(fig, title, f"One dot per state (n = {len(frame)})."
                              f"Dashed line = linear fit; labeled states are the biggest outliers.", height=520)
