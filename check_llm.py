"""Find out which AI models work for your key. Run:  python check_llm.py
Lists the models your key can use, then tests the likely candidates with and without tools.
It never prints your key."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()
import openai  # noqa: E402

from llm import OPENAI_TOOLS, make_client, settings  # noqa: E402

cfg = settings()
print("Provider:", cfg["provider"], "| URL:", cfg["base_url"])
client = make_client(os.getenv("LLM_API_KEY"))
if client is None:
    print("No LLM_API_KEY found. Check your .env file.")
    sys.exit(1)

SKIP = ("whisper", "guard", "tts", "orpheus", "playai", "embed", "safeguard")
CANDIDATES = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

try:
    available = sorted(m.id for m in client.models.list().data if not any(s in m.id.lower() for s in SKIP))
    print("\nModels your key can use:")
    for m in available:
        print("  ", m)
except openai.APIError as e:
    print(f"\nCould not list models ({type(e).__name__}); testing the candidates anyway.")
    available = None

to_test = [os.getenv("LLM_MODEL")] if os.getenv("LLM_MODEL") else \
    [m for m in CANDIDATES if available is None or m in available]
if not to_test and available:
    to_test = available[:3]


def attempt(model, label, **kw):
    try:
        r = client.chat.completions.create(model=model, max_tokens=300, **kw)
        m = r.choices[0].message
        shown = (m.content or "").strip()[:50] or f"tool call {[c.function.name for c in (m.tool_calls or [])]}"
        return True, shown
    except openai.APIStatusError as e:
        return False, f"HTTP {e.status_code}: {str(e)[:160]}"
    except openai.APIError as e:
        return False, f"{type(e).__name__}: {repr(e.__cause__)[:120]}"


print("\nTesting:")
good = []
for model in to_test:
    ok1, t1 = attempt(model, "plain", messages=[{"role": "user", "content": "Say hi in 3 words."}])
    ok2, t2 = attempt(model, "tools", tools=OPENAI_TOOLS, messages=[
        {"role": "user", "content": "Which states had the largest rise in unemployment during 2020?"}])
    print(f"  {model}\n     plain: {'OK  ' if ok1 else 'FAIL'} {t1}\n     tools: {'OK  ' if ok2 else 'FAIL'} {t2}")
    if ok1 and ok2:
        good.append(model)

print("\nResult:", f"put LLM_MODEL={good[0]} in your .env" if good else "no working model found, paste this output")
