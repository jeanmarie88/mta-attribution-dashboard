# Multi-touch attribution dashboard.
# Anonymised case study. Reads pre-aggregated CSVs from ./data - no user-level data.
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

TITLE = "Multi-Touch Attribution - Digital Channel Credit Allocation"
SUBTITLE = "Heuristic and data-driven attribution across the digital customer journey"
DATA = Path(__file__).parent / "data"

MODELS = ["First touch", "Last touch", "Time decay", "Markov", "Shapley"]

# Colour-blind-safe categorical palette, ordered for adjacent-series contrast.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C"]
GRID = "rgba(128,128,128,0.18)"

NOTES = {
    "overview": (
        "Eight digital channels, a seven-day observation window per user, and a 3.4% "
        "conversion rate. The window is the single most important caveat on everything "
        "below: touchpoints that fall outside it do not exist as far as any of these "
        "models are concerned, which systematically flatters channels appearing close to "
        "conversion and understates upper-funnel activity."
    ),
    "models": (
        "Heuristics encode an assumption about where credit belongs; Markov and Shapley "
        "learn it from the full interaction sequence across both converters and "
        "non-converters. The non-converting paths are the point - they are the "
        "counterfactual evidence the heuristics discard. Where the two families disagree "
        "is where the heuristics were wrong, and the size of the gap is the value of the "
        "extra complexity."
    ),
    "journeys": (
        "Credit allocation only makes sense against journey shape. Channels that appear "
        "predominantly at the opening of a path are doing discovery work that last touch "
        "cannot see, and the models that re-weight them are responding to real structure "
        "rather than noise."
    ),
    "efficiency": (
        "Contribution and efficiency are different questions. A channel can carry little "
        "absolute credit and still convert the users it reaches at a high rate - worth "
        "more budget, not less. Without cost data none of this answers the actual "
        "decision, which is where the next euro goes."
    ),
}


st.set_page_config(page_title="Attribution Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def load():
    return {p.stem: pd.read_csv(p) for p in DATA.glob("*.csv")}


d = load()
channels = d["channels"]
headline = d["headline"].iloc[0]
order = channels.sort_values("Shapley", ascending=False).channel.tolist()
CMAP = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}

# Everything below is derived from the loaded aggregates, so the commentary
# re-states itself when the underlying data changes.
n_channels = int(headline.channels)
conv_rate = float(headline.conversion_rate_pct)
window_days = int(headline.observed_window_days)
top_channel = order[0]
top_share = float(channels.loc[channels.channel == top_channel, "Shapley"].iloc[0])
gap = (channels.set_index("channel")["Shapley"] - channels.set_index("channel")["Last touch"])
gained = gap.idxmax()
lost = gap.idxmin()
best_rate = channels.sort_values("conv_rate_pct", ascending=False).iloc[0]
biggest_volume = channels.sort_values("share_of_touchpoints_pct", ascending=False).iloc[0]
nonconv = d["path_len"].groupby("converted").users.sum()
imbalance = float(nonconv.get(False, 0)) / max(float(nonconv.get(True, 1)), 1)

NOTES = {
    "overview": (
        f"{n_channels} digital channels, an observation window of up to {window_days} days "
        f"per user, and a {conv_rate:.2f}% conversion rate on the data currently loaded. "
        "The window is the single most important caveat on everything below: touchpoints "
        "falling outside it do not exist as far as any of these models are concerned, "
        "which systematically flatters channels appearing close to conversion and "
        "understates upper-funnel activity. Widen the window and these shares move."
    ),
    "models": (
        "Heuristics encode an assumption about where credit belongs; Markov and Shapley "
        "learn it from the full interaction sequence across both converters and "
        "non-converters. The non-converting paths are the point: they are the "
        "counterfactual evidence the heuristics discard. On the current data the widest "
        f"disagreement is {gained}, which gains {gap.max():+.1f} points of credit moving "
        f"from last touch to Shapley, while {lost} gives up {gap.min():.1f}. The size of "
        "that gap is what the extra complexity is buying."
    ),
    "journeys": (
        "Credit allocation only makes sense against journey shape. Channels appearing "
        "predominantly at the opening of a path are doing discovery work that last touch "
        "cannot see, and the models that re-weight them are responding to real structure "
        f"rather than noise. Converting journeys currently run to a median of "
        f"{headline.median_touches_converted:.0f} touchpoints over "
        f"{headline.median_span_days_converted:.1f} days."
    ),
    "efficiency": (
        "Contribution and efficiency are different questions. A channel can carry little "
        "absolute credit and still convert the users it reaches at a high rate, which "
        "argues for more budget rather than less. On the current data "
        f"{best_rate.channel} converts {best_rate.conv_rate_pct:.1f}% of the users it "
        f"reaches while carrying {best_rate.Shapley:.1f}% of total credit. Without cost "
        "data none of this answers the actual decision, which is where the next euro goes."
    ),
}

def style(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------- KPI cards
CARD_COLOURS = {
    "users": "#4C72B0",
    "touchpoints": "#8172B3",
    "conversions": "#55A868",
    "rate": "#DD8452",
    "median": "#0F8C8C",
}


def card(col, label, value, colour, footnote=""):
    col.markdown(
        f'''
        <div style="
            border: 1px solid {colour}55;
            border-left: 5px solid {colour};
            border-radius: 10px;
            padding: 0.85rem 1rem;
            background: linear-gradient(135deg, {colour}22, {colour}0D);
            height: 100%;
        ">
          <div style="font-size:0.78rem; letter-spacing:.04em; text-transform:uppercase;
                      opacity:.75; margin-bottom:.25rem;">{label}</div>
          <div style="font-size:1.85rem; font-weight:650; line-height:1.15;
                      color:{colour};">{value}</div>
          <div style="font-size:0.75rem; opacity:.65; margin-top:.2rem;">{footnote}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- header
st.title(TITLE)
st.caption(SUBTITLE)

k = st.columns(5)
card(k[0], "Users", f"{int(headline.users):,}", CARD_COLOURS["users"],
     f"{window_days}-day observation window")
card(k[1], "Touchpoints", f"{int(headline.touchpoints):,}", CARD_COLOURS["touchpoints"],
     f"across {n_channels} channels")
card(k[2], "Conversions", f"{int(headline.conversions):,}", CARD_COLOURS["conversions"],
     f"{imbalance:.0f}:1 non-converters to converters")
card(k[3], "Conversion rate", f"{conv_rate:.2f}%", CARD_COLOURS["rate"],
     "share of observed users converting")
card(k[4], "Median touches", f"{headline.median_touches_converted:.0f}",
     CARD_COLOURS["median"], "converting journeys")

st.caption(
    f"Observation window {headline.window_start} to {headline.window_end} · "
    f"{biggest_volume.channel} carries the most touchpoint volume "
    f"({biggest_volume.share_of_touchpoints_pct:.0f}%); {top_channel} takes the largest "
    f"share of Shapley credit ({top_share:.1f}%)."
)
st.divider()

tab_overview, tab_models, tab_journeys, tab_method = st.tabs(
    ["Overview", "Model comparison", "Journey structure", "Method & caveats"]
)

# ---------------------------------------------------------------- overview
with tab_overview:
    st.info(NOTES["overview"])
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Credit by channel")
        model = st.radio("Attribution model", MODELS, index=4, horizontal=True)
        c = channels.sort_values(model, ascending=True)
        fig = go.Figure(
            go.Bar(
                x=c[model], y=c.channel, orientation="h",
                marker_color=[CMAP[ch] for ch in c.channel],
                text=[f"{v:.1f}%" for v in c[model]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.2f}% of credit<extra></extra>",
            )
        )
        fig.update_layout(title=f"{model} - share of attributed conversions")
        fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style(fig, 460, legend=False), width="stretch")

    with right:
        st.subheader("Volume vs. credit")
        fig = go.Figure()
        for ch in order:
            row = channels[channels.channel == ch].iloc[0]
            fig.add_trace(go.Scatter(
                x=[row.share_of_touchpoints_pct], y=[row.Shapley],
                mode="markers+text", name=ch, text=[ch], textposition="top center",
                marker=dict(size=14, color=CMAP[ch]),
                hovertemplate=(f"<b>{ch}</b><br>%{{x:.1f}}% of touchpoints"
                               f"<br>%{{y:.1f}}% of credit<extra></extra>"),
            ))
        lim = max(channels.share_of_touchpoints_pct.max(), channels.Shapley.max()) * 1.15
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                      line=dict(dash="dot", color="rgba(128,128,128,0.5)"))
        fig.update_layout(title="Shapley credit against share of touchpoints")
        fig.update_xaxes(title="Share of all touchpoints", ticksuffix="%", range=[0, lim])
        fig.update_yaxes(title="Shapley credit", ticksuffix="%", range=[0, lim])
        st.plotly_chart(style(fig, 460, legend=False), width="stretch")
        st.caption("Points above the dotted line earn more credit than their raw volume implies.")

    st.subheader("Efficiency")
    st.caption(NOTES["efficiency"])
    e = channels.sort_values("conv_rate_pct", ascending=False)
    fig = go.Figure(go.Bar(
        x=e.channel, y=e.conv_rate_pct,
        marker_color=[CMAP[ch] for ch in e.channel],
        text=[f"{v:.1f}%" for v in e.conv_rate_pct], textposition="outside",
        customdata=e[["users_reached", "Shapley"]],
        hovertemplate=("%{x}<br>%{y:.2f}% of reached users convert"
                       "<br>%{customdata[0]:,} users reached"
                       "<br>%{customdata[1]:.1f}% of credit<extra></extra>"),
    ))
    fig.update_layout(title="Conversion rate among users each channel reached")
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(style(fig, 380, legend=False), width="stretch")

# ---------------------------------------------------------------- models
with tab_models:
    st.info(NOTES["models"])

    long = channels.melt(id_vars="channel", value_vars=MODELS,
                         var_name="model", value_name="credit")
    long["channel"] = pd.Categorical(long.channel, order, ordered=True)
    fig = px.bar(long.sort_values("channel"), x="model", y="credit", color="channel",
                 barmode="group", color_discrete_map=CMAP, category_orders={"model": MODELS})
    fig.update_layout(title=f"Credit allocation across all {len(MODELS)} models")
    fig.update_yaxes(title="Share of credit", ticksuffix="%")
    fig.update_xaxes(title=None)
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:.2f}%<extra></extra>")
    st.plotly_chart(style(fig, 460), width="stretch")

    st.subheader("What the data-driven models change")
    base = st.selectbox("Compare against", ["Last touch", "First touch", "Time decay"], index=0)
    target = st.selectbox("Data-driven model", ["Shapley", "Markov"], index=0)
    delta = channels[["channel"]].copy()
    delta["shift"] = channels[target] - channels[base]
    delta = delta.sort_values("shift")
    fig = go.Figure(go.Bar(
        x=delta["shift"], y=delta.channel, orientation="h",
        marker_color=["#C44E52" if v < 0 else "#55A868" for v in delta["shift"]],
        text=[f"{v:+.1f} pp" for v in delta["shift"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.2f} percentage points<extra></extra>",
    ))
    fig.update_layout(title=f"{target} minus {base} (percentage points of credit)")
    fig.add_vline(x=0, line_color="rgba(128,128,128,0.6)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    st.plotly_chart(style(fig, 400, legend=False), width="stretch")
    st.caption(
        "Green channels are under-credited by the heuristic; red are over-credited. "
        "Directionally, upper-funnel channels should gain under a data-driven model - "
        "a sharp deviation without a behavioural explanation is a prompt to review "
        "assumptions, not a discovery."
    )

    st.subheader("Model table")
    show = channels[["channel"] + MODELS + ["users_reached", "conv_rate_pct"]].copy()
    show = show.sort_values("Shapley", ascending=False)
    st.dataframe(
        show.style.format({**{m: "{:.2f}%" for m in MODELS},
                           "conv_rate_pct": "{:.2f}%", "users_reached": "{:,.0f}"})
            .background_gradient(cmap="Blues", subset=MODELS),
        width="stretch", hide_index=True,
    )

# ---------------------------------------------------------------- journeys
with tab_journeys:
    st.info(NOTES["journeys"])
    left, right = st.columns(2)

    with left:
        st.subheader("Where channels sit in the path")
        f = d["funnel"].copy()
        f["touchpoint"] = pd.Categorical(f.touchpoint, order[::-1], ordered=True)
        fig = px.bar(f.sort_values("touchpoint"), x="pct", y="touchpoint", color="stage",
                     orientation="h", barmode="stack",
                     color_discrete_sequence=["#4C72B0", "#8172B3", "#DD8452"],
                     category_orders={"stage": ["Opening", "Middle", "Closing"]})
        fig.update_layout(title="Position within the journey")
        fig.update_xaxes(ticksuffix="%", title=None, range=[0, 100])
        fig.update_yaxes(title=None)
        fig.update_traces(hovertemplate="%{y} - %{fullData.name}: %{x:.1f}%<extra></extra>")
        st.plotly_chart(style(fig, 440), width="stretch")

    with right:
        st.subheader("Journey length")
        p = d["path_len"].copy()
        p["group"] = p.converted.map({True: "Converted", False: "Did not convert"})
        fig = px.bar(p, x="bucket", y="users", color="group", barmode="group",
                     color_discrete_map={"Converted": "#55A868",
                                         "Did not convert": "#8C8C8C"})
        fig.update_layout(title="Users by number of touchpoints")
        fig.update_yaxes(title="Users", type="log")
        fig.update_xaxes(title="Touchpoints in journey")
        st.plotly_chart(style(fig, 440), width="stretch")
        st.caption(f"Log scale. Non-converters currently outnumber converters by roughly {imbalance:.0f} to 1.")

    tp = d["top_paths"]
    st.subheader("Most common closing sequences")
    st.caption(f"Last three touchpoints before conversion, top {len(tp)} sequences "
               "in the loaded data.")
    fig = go.Figure(go.Bar(
        x=tp.conversions[::-1], y=tp.path[::-1], orientation="h",
        marker_color="#4C72B0",
        hovertemplate="%{y}<br>%{x} conversions<extra></extra>",
    ))
    fig.update_layout(title=None)
    fig.update_xaxes(title="Conversions", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None)
    st.plotly_chart(style(fig, 520, legend=False), width="stretch")

    st.subheader("Touchpoint volume over time")
    daily = d["daily"].copy()
    daily["date"] = pd.to_datetime(daily.date)
    picked = st.multiselect("Channels", order, default=order)
    dd = daily[daily.touchpoint.isin(picked)]
    fig = px.line(dd, x="date", y="touchpoints", color="touchpoint",
                  color_discrete_map=CMAP, category_orders={"touchpoint": order})
    fig.update_traces(line_width=1.6, hovertemplate="%{x|%d %b}: %{y}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Touchpoints per day")
    st.plotly_chart(style(fig, 400), width="stretch")

# ---------------------------------------------------------------- method
with tab_method:
    st.markdown(
        f'''
### Models

| Model | What it assumes | What it is good for |
|---|---|---|
| **First touch** | All credit to the journey's opening touchpoint | Reading discovery |
| **Last touch** | All credit to the closing touchpoint | Reading what closes |
| **Time decay** | Credit decays with distance from conversion | A defensible middle position |
| **Markov** | Credit is the removal effect - the drop in conversions if the channel disappeared | Channel interactions and dependencies |
| **Shapley** | Credit is the average marginal contribution across all channel coalitions | A fair, axiomatic allocation |

The heuristics are a baseline, not a straw man: they are the models stakeholders already
carry in their heads, so knowing what the data-driven output disagrees with - and by how
much - is part of the deliverable.

### Known biases

- **Window truncation.** The observation window in the loaded data runs to at most
  {window_days} days per user, which cuts real journeys short. Upper-funnel activity
  outside it is invisible, so channels near conversion are over-weighted. Every figure in
  this dashboard inherits that bias.
- **Class imbalance.** A {conv_rate:.2f}% conversion rate leaves
  {int(headline.conversions):,} positive examples against roughly {imbalance:.0f} times as
  many non-converters, which biases results toward high-volume channels and leaves
  low-volume channels statistically unstable.
- **Omitted variables.** No campaign ID, device, geography or cost, so it is impossible to
  say which specific activity worked, how behaviour varies by segment, or what any of it
  costs.
- **No engagement metrics.** Passive exposure and active engagement are
  indistinguishable, so intent cannot be assessed.

Most of these are data problems rather than modelling problems - enriching the dataset
moves the result far more than changing the estimator.

### Validation

Data-driven output was benchmarked against the heuristics, attribution weights were
bootstrapped for confidence intervals (tight, symmetric, near-zero bias), and the lookback
window and decay half-life were varied to check that channel rankings survive the
parameter choice.

### What attribution cannot do

Attribution describes credit allocation across observed paths. It is not a causal claim.
If a model estimates a given drop in conversions when a channel is removed, that is a
*testable prediction* - a geo holdout or an A/B test is what establishes whether it
materialises. In a full measurement stack, marketing mix modelling supplies macro-level
impact including offline, multi-touch attribution explains micro-level digital pathways,
and incrementality testing provides causal ground truth. Calibrating the three against
each other is what makes the stack trustworthy.

---

*Anonymised case study. Every figure on this page is recomputed from the aggregate files
in `data/`, so refreshing those files refreshes the dashboard and this commentary.
Currently loaded: {int(headline.users):,} users, {int(headline.touchpoints):,} touchpoints,
{int(headline.conversions):,} conversions, {headline.window_start} to {headline.window_end}.*
        '''
    )

st.divider()
st.caption("Built with Streamlit · aggregated results only, no user-level data included.")
