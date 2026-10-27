# EconSight AI

A dashboard and analysis of unemployment and labor force participation in all 50 US states (Jan 2020 to Dec 2021), with an AI assistant that answers plain-English questions from the data.

Live demo: add your Streamlit link here

A falling unemployment rate can mean people found jobs, or that they stopped looking for work. So unemployment alone can make a recovery look better than it was.

This project checks both unemployment and labor force participation (the share of adults who are working or looking for work) in every state to answer:

Which states were hit hardest, and when?
How long did each state take to recover, and which had not by December 2021?
Did participation recover along with unemployment?
Is a bigger jump in unemployment linked to a bigger drop in participation?

## Key findings
The shock was sharp and uneven: when states peaked, the jump in the average rate, and the hardest-hit and least-hit states.

Unemployment mostly recovered, but slowly and not everywhere: the 47 of 50 recovered states, the median of 15 months, the fastest and slowest states, and the three that hadn’t recovered.

Participation did not recover, even where unemployment did.

## Data source
Retrieved from the FRED API (Federal Reserve Bank of St. Louis), sourced from the U.S. Bureau of Labor Statistics.

## The AI analyst

Users can ask questions such as *"Which states had the largest rise in unemployment during 2020?"* or *"How fast did Texas recover?"*

**How it works:** the model never calculates or remembers numbers. Python runs tools on the data, and the model explains the result. Every answer has a **"Data used"** panel showing the exact tool calls.

**Guardrails:** scope limited to this dataset; inputs validated (states, months, metrics) with errors returned to the model; limits on question length and questions per session; friendly messages when the AI service fails; API key kept out of the code.

**Providers:** any OpenAI-compatible API (Groq, Gemini, OpenRouter, or Ollama locally), chosen with `LLM_PROVIDER` in `.env`.

**Tip:** ask one question at a time. Very broad multi-part questions (for example, "which state had the highest peak, fastest recovery and lowest participation?") can run out of tool attempts and fail.


## Run it yourself

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys (never commit .env)
python fetch_data.py          # optional: refresh data from FRED (needs FRED_API_KEY)
streamlit run app.py
```

The `data/` folder is included, so the dashboard runs without a FRED key. The AI features need `LLM_PROVIDER` and `LLM_API_KEY`.

| File | Purpose |
|---|---|
| `fetch_data.py` | Pulls the 50 state series from FRED into `data/` |
| `tools.py` | Data loading, summary statistics, and the tools the AI can call |
| `charts.py` | All Plotly charts |
| `llm.py` | The AI analyst: prompt, tool-use loop, provider settings |
| `app.py` | The Streamlit dashboard |
| `notebook/` | The original exploratory notebook |


## Free AI provider
Any OpenAI-compatible API works. Set `LLM_PROVIDER` in `.env` and get a free key from that provider:
| `LLM_PROVIDER` | Where to get a key | Notes |
|---|---|---|
| `groq` (default) | console.groq.com | Fast, free tier with rate limits |
| `gemini` | aistudio.google.com | Free tier through Google AI Studio |
| `openrouter` | openrouter.ai | Pick a model ending in `:free` |
| `ollama` | none needed | Runs on your PC; install Ollama and run `ollama pull llama3.1` |

Free tiers have daily and per-minute limits that change, and model names get retired. If you get errors,
set `LLM_MODEL` in `.env` to a current tool-calling model from the provider's list.


## Limitations
State estimates are model-based and revised yearly. Scope is 50 states, Jan 2020 to Dec 2021. The AI
describes what happened, not why.
