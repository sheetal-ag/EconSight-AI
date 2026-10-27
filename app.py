import json
import os

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import charts
from llm import answer, generate_insights, make_client
from tools import LaborData, label, load_data, load_metadata

load_dotenv()
DATA_DIR = os.getenv("ECON_DATA_DIR", "data")
MAX_QUESTION_CHARS = 500
MAX_QUESTIONS_PER_SESSION = 20
EXAMPLES = ["Which states had the largest rise in unemployment during 2020?",
            "How did participation change from January to December 2020?",
            "How fast did Texas recover?",
            "Is there a relationship between unemployment and participation?"]

st.set_page_config(page_title="EconSight AI", page_icon="📈", layout="wide")


@st.cache_resource
def get_data(folder):
    return LaborData(*load_data(folder))


try:
    data = get_data(DATA_DIR)
except FileNotFoundError:
    st.error("Data files not found. Run `python fetch_data.py` first (see README).")
    st.stop()
meta = load_metadata(DATA_DIR)


def get_api_key():
    key = os.getenv("LLM_API_KEY")
    if not key:
        try:
            key = st.secrets.get("LLM_API_KEY")
        except Exception:
            key = None
    return key


# ---------- sidebar filters ----------
months = list(data.unemp.index)
fmt = lambda d: d.strftime("%b %Y")
with st.sidebar:
    st.header("Filters")
    metric = st.radio("Indicator", list(charts.LABEL), format_func=charts.LABEL.get)
    start, end = st.select_slider("Date range", options=months, value=(months[0], months[-1]), format_func=fmt)
    defaults = [s for s in ["California", "Nevada", "Texas"] if s in data.states]
    picked = st.multiselect("States (up to 6)", data.states, default=defaults, max_selections=6)
    mode = st.radio("Trend view", ["Level", f"Change since {label(charts.BASELINE)}"],
                    help="Change view lines every state up at zero in Feb 2020, which makes states easy to compare.")
    st.divider()
    st.caption("Impact, recovery and relationship charts always use the full Jan 2020 to Dec 2021 period.")

# ---------- header + KPIs ----------
st.title("EconSight AI")
st.caption("US state labor markets during COVID-19, with an AI analyst that answers questions from the data.")
h = data.highlights()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Highest unemployment peak", f"{h['highest_peak']['rate']}%")
k1.caption(f"{h['highest_peak']['state']}, {h['highest_peak']['month']}")
if "fastest_recovery" in h:
    k2.metric("Fastest recovery", h["fastest_recovery"]["state"],
              help="Months until unemployment was within 1 point of its Feb 2020 level.")
    k2.caption(f"{h['fastest_recovery']['months']} months from peak")
k3.metric("Not recovered by Dec 2021", f"{len(h['not_recovered_by_end'])} states")
k4.metric("Participation still below Feb 2020", f"{h['participation_still_below_baseline_count']} states",
          help="More than 0.5 points below the Feb 2020 level in the latest month.")

# ---------- charts ----------
frame = data.frames[metric]
tabs = st.tabs(["Trends", "Impact & recovery", "Heatmap", "Map", "Relationship", "Data"])

with tabs[0]:
    if picked:
        st.plotly_chart(charts.trend_chart(frame, picked, metric, mode, start, end), width="stretch")
    else:
        st.info("Pick at least one state in the sidebar.")

with tabs[1]:
    st.plotly_chart(charts.impact_dumbbell(data.summary), width="stretch")
    st.plotly_chart(charts.recovery_bars(data.summary), width="stretch")

with tabs[2]:
    st.plotly_chart(charts.heatmap(frame, metric, start, end), width="stretch")

with tabs[3]:
    in_range = [m for m in months if start <= m <= end]
    default_month = pd.Timestamp("2020-04-01") if pd.Timestamp("2020-04-01") in in_range else in_range[0]
    month = st.select_slider("Month", options=in_range, value=default_month, format_func=fmt) if len(in_range) > 1 else in_range[0]
    st.plotly_chart(charts.us_map(frame, metric, month), width="stretch")

with tabs[4]:
    st.plotly_chart(charts.relationship_scatter(data.relationship_frame()), width="stretch")

with tabs[5]:
    table = frame.loc[start:end].rename_axis("Month")
    st.dataframe(table.style.format("{:.1f}"), width="stretch", height=380)
    st.download_button("Download CSV", table.to_csv().encode(), f"{metric}_{start:%Y%m}_{end:%Y%m}.csv", "text/csv")
    with st.expander("About this data"):
        st.markdown(
            "- **Source:** U.S. Bureau of Labor Statistics, retrieved through the FRED API (St. Louis Fed).\n"
            "- **Series:** state unemployment rate (`<state>UR`) and labor force participation rate (`LBSSA<fips>`), "
            "monthly, seasonally adjusted, in percent.\n"
            f"- **Coverage:** 50 states, {label(data.first)} to {label(data.last)}. DC and Puerto Rico are excluded.\n"
            "- **Recovered** means unemployment is within 1 point of its Feb 2020 level, counted from the peak month.\n"
            "- State estimates are model-based and revised every year, so values can differ slightly from the "
            "originally published releases.\n"
            + (f"- **Data retrieved:** {meta['retrieved']}\n" if meta.get("retrieved") else ""))

# ---------- AI analyst ----------
client = make_client(get_api_key())
st.divider()
st.subheader("AI analyst")
if client is None:
    st.info("Set LLM_API_KEY in your .env file (or use LLM_PROVIDER=ollama) to turn on AI insights and the chat.")
else:
    if st.button("Generate key insights"):
        with st.spinner("Writing insights..."):
            try:
                st.session_state.insights = generate_insights(client, data)
            except openai.RateLimitError:
                st.session_state.insights = "The free AI tier is busy or at its limit. Wait a minute and try again."
            except openai.APIError:
                st.session_state.insights = "Sorry, insights are unavailable right now."
    if "insights" in st.session_state:
        st.markdown(st.session_state.insights)

st.caption("Answers are calculated from the dataset by Python tools; the AI only explains the results. "
           "Check important figures against FRED.")
cols = st.columns(len(EXAMPLES))
for col, text in zip(cols, EXAMPLES):
    if col.button(text, disabled=client is None, width="stretch"):
        st.session_state.pending = text

if "chat" not in st.session_state:
    st.session_state.chat = []
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("trace"):
            with st.expander("Data used"):
                st.code(json.dumps(m["trace"], indent=2), language="json")

question = st.chat_input("Ask about unemployment or participation...", disabled=client is None)
question = question or st.session_state.pop("pending", None)
if question:
    asked = sum(m["role"] == "user" for m in st.session_state.chat)
    if len(question) > MAX_QUESTION_CHARS:
        st.warning(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    elif asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit reached for this session. Refresh the page to start again.")
    else:
        st.chat_message("user").write(question)
        history = st.session_state.chat + [{"role": "user", "content": question}]
        with st.chat_message("assistant"):
            with st.spinner("Checking the data..."):
                try:
                    reply, trace = answer(client, data, history)
                except openai.RateLimitError:
                    reply, trace = "The free AI tier is busy or at its limit. Wait a minute and try again.", []
                except openai.APIError:
                    reply, trace = "Sorry, I can't reach the AI service right now. Please try again.", []
            st.write(reply)
            if trace:
                with st.expander("Data used"):
                    st.code(json.dumps(trace, indent=2), language="json")
        st.session_state.chat = history + [{"role": "assistant", "content": reply, "trace": trace}]
