# 🛠️ STEPS — run the Insight Agent locally & deploy

## 1. Get the code
```bash
git clone https://github.com/AnkitSaxena-AI/insight-agent.git
cd insight-agent
```

## 2. Create an environment & install
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Get a free LLM key (Groq — recommended)
1. Sign up at https://console.groq.com (no credit card).
2. Create a key at https://console.groq.com/keys.
3. Copy `.env.example` to `.env` and paste your key:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
   *(Prefer Gemini? `pip install -r requirements-optional.txt`, set `LLM_PROVIDER=gemini`
   and `GOOGLE_API_KEY=...` from https://aistudio.google.com/app/apikey.)*

## 4. Run the app
```bash
streamlit run app/app.py
```
Pick a bundled sample (or upload a CSV), type a question, hit **Analyze**, and watch
the agent plan → code → run → self-correct → answer.

## 5. Run the offline tests (no key needed)
```bash
pip install -r requirements-dev.txt
pytest -q
```

## 6. Deploy on Streamlit Community Cloud
1. Push to GitHub.
2. Go to https://share.streamlit.io → **New app** → pick the repo → main file `app/app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Deploy. Add the live URL to the README badge.
