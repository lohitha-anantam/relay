# The Relay — LLM Pipeline Performance & Reliability Lab

A Streamlit implementation scaffold for the supplied Relay assignment. It includes a multi-step paper QA pipeline, exact-version arXiv PDF retrieval, Groq API integration, a configurable stand-in service, load experiments and raw CSV logging.

## 1. Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Use Python 3.11 or 3.12 for easiest deployment compatibility.

## 2. Prepare the library

In the app sidebar click **Fetch / verify the 3 required papers**. The app fetches these exact versions:
- `1706.03762v7` — Attention Is All You Need
- `2410.00037v1` — Moshi: a speech-text foundation model for real-time dialogue
- `2507.07955v2` — Dynamic Chunking for End-to-End Hierarchical Sequence Modeling

The PDFs are stored locally under `data/papers/` and are excluded from Git by `.gitignore`. The app can re-fetch them on first use.

## 3. Groq key

Create your own Groq account and API key. In the app, select **Real Groq API** and paste the key in the sidebar, or set `GROQ_API_KEY` in your environment. Never commit the key or include it in the submission. Select a model available to your Groq account. Model availability and limits change; check current Groq documentation and response headers when you run the experiment.

For Streamlit Community Cloud, add `GROQ_API_KEY` to App settings → Secrets, then modify/use the environment variable path supported by the deployment. Do not hard-code it.

## 4. Experiments

- **Run pipeline:** one question; records a raw row.
- **Load experiments:** steady, burst, sustained overload, per-minute limit halved, and daily-budget-exhausted stand-in scenarios.
- **Retrieval evaluation:** three configurations (`small_fast`, `balanced`, `large_context`) across at least 15 questions.
- **Results & submission:** inspect and download `results.csv`; tune stand-in latency, jitter, failure probability, rate limiting and load slowdown.

The stand-in is intentionally controllable. Its defaults are placeholders, not claims about Groq. After collecting real Groq measurements, tune its latency/jitter/failure settings to approximate your observed distribution and document where it differs.

## 5. Required raw output

`results.csv` has the assignment's 12 named columns:
`request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`.

The PDF brief calls this "these columns" and enumerates 12 fields. It says one row per request. The application records provider failures as rows too.

**Important:** If an API call fails, the app records the failure row rather than silently losing it. Review the status and answer_grade values before submission. Do not label stand-in results as real.

## 6. Important limitations to address before submission

This is a runnable engineering baseline, not fabricated completed research:
1. Groq quota values (RPM, TPM, RPD, TPD, concurrency, reset timing) must be measured/verified from your account's live response headers and current provider documentation. The app does not guess them.
2. The local retrieval is lexical overlap ranking, not a semantic embedding retriever. This is useful as a transparent baseline but can miss paraphrases and dense technical matches.
3. The lightweight answer-grade heuristic is not a substitute for manual evaluation. Review at least 15 answers against source pages and correct grades where needed.
4. The stand-in is a controllable approximation, not a faithful Groq replica. It does not reproduce provider scheduling, token accounting, model-specific queueing, or exact quota windows.
5. Streaming first-token timing is measured from the first streamed content token. Usage may be absent from some streaming responses; inspect raw response headers and reconcile token counts where needed.
6. The current multi-step workflow uses separate plan, draft and verification calls, so it is slower and more quota-expensive than a single-call design. This is a deliberate reliability/quality tradeoff to measure.

## 7. Render deployment

Set build command to `pip install -r requirements.txt` and start command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` on a compatible host. Streamlit Community Cloud uses its normal app deployment flow. Do not commit PDFs, secrets, or your real results file unless you intend to share them.

## 8. Submission checklist

- [ ] Fetch and verify the three exact paper versions.
- [ ] Make a modest real Groq sample; capture headers, latency distribution, 429 body, and recovery.
- [ ] Save raw real request rows.
- [ ] Run stand-in steady, burst and sustained overload tests.
- [ ] Run per-minute-halved and daily-budget-exhausted stand-in cases.
- [ ] Evaluate at least 15 self-authored questions and manually audit citations/refusals.
- [ ] Compare all three retrieval settings.
- [ ] Fill the assignment tables from raw measurements; write "not measured" where applicable.
- [ ] Write the two-page reflection with the exact headings in the brief.
- [ ] Include your actual assistant conversation history/prompts and explain your own decisions.

The brief asks for 13? No: it lists 12 CSV fields. Always use the exact field list from page 4 of the brief.
