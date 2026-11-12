import streamlit as st
import pandas as pd
import os, json, time, uuid, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from services.pipeline import RelayPipeline, QUESTIONS, SETTINGS, PipelineRecordedError
from services.providers import GroqProvider, StandInProvider, ProviderError
from services.results import append_result, load_results, RESULTS_PATH
from services.library import ensure_library, library_status

st.set_page_config(page_title="The Relay | LLM Reliability Lab", page_icon="🔁", layout="wide")
st.title("🔁 The Relay")
st.caption("LLM pipeline performance, retrieval quality and reliability laboratory")

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "data" / "config.json"

def read_config():
    if CONFIG_PATH.exists():
        try: return json.loads(CONFIG_PATH.read_text())
        except Exception: pass
    return {"groq_model":"llama-3.1-8b-instant", "standin":{"base_latency_ms":900,"jitter_ms":250,"failure_rate":0.03,"rate_limit_every":0,"load_slowdown_ms":120,"tokens_per_minute":12000,"stream_chunk_chars":18}}

def save_config(c):
    CONFIG_PATH.write_text(json.dumps(c, indent=2))

cfg = read_config()
with st.sidebar:
    st.header("Provider")
    provider_mode = st.radio("Execution mode", ["Stand-in (safe for load tests)", "Real Groq API"], index=0)
    groq_key = st.text_input("Groq API key", type="password", help="Stored only in this running session; never include it in your submission.")
    groq_model = st.text_input("Groq model", value=cfg.get("groq_model","llama-3.1-8b-instant"))
    st.caption("Use a small number of real requests first. Use the stand-in for high-load experiments.")
    st.divider()
    st.header("Library")
    if st.button("Fetch / verify the 3 required papers", use_container_width=True):
        with st.spinner("Fetching exact arXiv versions..."):
            try:
                ensure_library()
                st.success("Library ready.")
            except Exception as e: st.error(str(e))
    st.write(library_status())

def make_provider(mode=None):
    mode = mode or provider_mode
    if mode == "Real Groq API":
        key = groq_key or os.getenv("GROQ_API_KEY","")
        if not key: raise ValueError("Enter your Groq API key in the sidebar or set GROQ_API_KEY.")
        return GroqProvider(key, groq_model)
    s = cfg["standin"]
    return StandInProvider(**s)

tab1, tab2, tab3, tab4 = st.tabs(["Run pipeline", "Load experiments", "Retrieval evaluation", "Results & submission"])

with tab1:
    st.subheader("Ask the research-paper assistant")
    question = st.text_area("Question", "What is the role of multi-head attention in the Transformer?", height=90)
    setting_name = st.selectbox("Retrieval setting", list(SETTINGS.keys()), index=1)
    if st.button("Run one request", type="primary"):
        try:
            p = make_provider()
            pipeline = RelayPipeline(p, setting_name)
            with st.spinner("Running classify → retrieve → plan → draft → verify..."):
                result = pipeline.run(question, scenario="single", source="real" if provider_mode=="Real Groq API" else "stand-in")
            append_result(result["row"])
            st.metric("Total time", f'{result["row"]["total_ms"]:.0f} ms')
            st.metric("Time to first token", f'{result["row"]["first_token_ms"]:.0f} ms')
            st.write("**Answer**")
            st.markdown(result["answer"])
            with st.expander("Retrieved evidence"):
                for item in result["retrieved"]:
                    st.markdown(f'**{item["paper"]} — page {item["page"]}** (score {item["score"]:.3f})')
                    st.write(item["text"])
            with st.expander("Execution trace"):
                st.json(result["trace"])
            st.success(f'Recorded request {result["row"]["request_id"]} in results.csv')
        except PipelineRecordedError as e:
            append_result(e.row)
            st.error(f"Request recorded as {e.row['status']}: {e}")
        except Exception as e: st.error(str(e))

with tab2:
    st.subheader("Controlled load experiments")
    st.warning("Load tests use the stand-in by default. Real Groq tests are deliberately capped; do not use this to intentionally exhaust your free-tier daily quota.")
    scenario = st.selectbox("Scenario", ["steady", "burst", "sustained_overload", "per_minute_limit_halved", "daily_budget_exhausted"])
    col1, col2, col3 = st.columns(3)
    n = col1.number_input("Requests", min_value=1, max_value=100, value=12)
    workers = col2.number_input("Concurrent workers", min_value=1, max_value=20, value=4)
    interval = col3.number_input("Launch interval (seconds)", min_value=0.0, max_value=10.0, value=0.15, step=0.05)
    setting = st.selectbox("Retrieval setting for load test", list(SETTINGS.keys()), index=1, key="loadsetting")
    use_real = st.checkbox("Use real Groq (small tests only)", value=False)
    if use_real and n > 10:
        st.error("For safety, real-provider batch is capped at 10 requests. Use stand-in for larger loads.")
    if st.button("Start experiment", type="primary"):
        if use_real and n > 10: st.stop()
        try:
            mode = "Real Groq API" if use_real else "Stand-in (safe for load tests)"
            p = make_provider(mode)
            if scenario == "per_minute_limit_halved" and not use_real:
                cfg["standin"]["rate_limit_every"] = max(2, int(cfg["standin"].get("rate_limit_every",0) or 6))
                save_config(cfg)
                p = StandInProvider(**cfg["standin"])
            if scenario == "daily_budget_exhausted" and not use_real:
                p = StandInProvider(**{**cfg["standin"], "daily_exhausted": True})
            questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(int(n))]
            pipeline_args = (p, setting)
            start = time.time()
            rows, errors = [], []
            progress = st.progress(0)
            status = st.empty()
            def task(ix_q):
                ix, q = ix_q
                if scenario == "burst": time.sleep(0.0)
                elif scenario == "steady": time.sleep(ix * float(interval))
                elif scenario == "sustained_overload": time.sleep(ix * min(float(interval),0.02))
                else: time.sleep(ix * float(interval))
                try:
                    out = RelayPipeline(*pipeline_args).run(q, scenario=scenario, source="real" if use_real else "stand-in")
                    append_result(out["row"])
                    return out["row"], None
                except PipelineRecordedError as ex:
                    append_result(ex.row)
                    return ex.row, str(ex)
                except Exception as ex:
                    return None, str(ex)
            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                futs = [pool.submit(task, x) for x in enumerate(questions)]
                for done, fut in enumerate(as_completed(futs), 1):
                    row, err = fut.result()
                    if row: rows.append(row)
                    if err: errors.append(err)
                    progress.progress(done/int(n))
                    status.write(f"Completed {done}/{int(n)}")
            elapsed = time.time()-start
            st.success(f"Experiment finished in {elapsed:.1f}s. Recorded {len(rows)} rows; {len(errors)} execution errors.")
            if errors: st.code("\n".join(errors[:5]))
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e: st.error(str(e))

with tab3:
    st.subheader("Retrieval quality versus cost")
    st.write("Runs a fixed question set against three retrieval configurations. Use the same provider mode for a fair comparison.")
    qcount = st.slider("Number of evaluation questions", 15, 30, 15)
    eval_real = st.checkbox("Use real Groq for retrieval evaluation", value=False)
    if st.button("Run retrieval comparison"):
        try:
            p = make_provider("Real Groq API" if eval_real else "Stand-in (safe for load tests)")
            eval_rows = []
            bar = st.progress(0)
            for i, sname in enumerate(SETTINGS):
                for j, q in enumerate(QUESTIONS[:qcount]):
                    out = RelayPipeline(p, sname).run(q, scenario="retrieval_eval", source="real" if eval_real else "stand-in")
                    append_result(out["row"])
                    eval_rows.append(out["row"])
                    bar.progress((i*qcount+j+1)/(len(SETTINGS)*qcount))
            df = pd.DataFrame(eval_rows)
            st.dataframe(df, use_container_width=True)
            st.success(f"Saved {len(eval_rows)} request rows. Filter results by retrieval_setting for Table 3.")
        except Exception as e: st.error(str(e))

with tab4:
    st.subheader("Raw results.csv")
    df = load_results()
    if df.empty:
        st.info("No requests recorded yet. Run the pipeline or an experiment first.")
    else:
        st.metric("Recorded requests", len(df))
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results.csv", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv", type="primary")
        st.subheader("Quick summary (computed from raw rows)")
        valid = df[df["status"]=="ok"]
        if not valid.empty:
            summary = valid.groupby(["scenario","source"]).agg(
                requests=("request_id","count"),
                first_token_p50=("first_token_ms","median"),
                first_token_p95=("first_token_ms",lambda x:x.quantile(.95)),
                total_p50=("total_ms","median"),
                total_p95=("total_ms",lambda x:x.quantile(.95)),
                tokens_per_answer=("tokens_out","mean")
            ).reset_index()
            st.dataframe(summary, use_container_width=True)
        st.caption("The assignment's tables 1, 2a, 2b and 3 must be computed from this raw file and your provider observations. Missing/unmeasured values must be labelled honestly.")
    st.download_button("Download current results.csv (even if empty)", (df.to_csv(index=False) if not df.empty else pd.DataFrame(columns=["request_id","scenario","source","start_time","first_token_ms","total_ms","status","retries","tokens_in","tokens_out","retrieval_setting","answer_grade"]).to_csv(index=False)).encode(), "results.csv", "text/csv")
    st.subheader("Stand-in calibration")
    s = cfg["standin"]
    c1,c2,c3 = st.columns(3)
    s["base_latency_ms"] = c1.number_input("Base latency (ms)", 100, 10000, int(s["base_latency_ms"]))
    s["jitter_ms"] = c2.number_input("Jitter (ms)", 0, 5000, int(s["jitter_ms"]))
    s["failure_rate"] = c3.slider("Failure probability", 0.0, 0.5, float(s["failure_rate"]))
    c4,c5,c6 = st.columns(3)
    s["load_slowdown_ms"] = c4.number_input("Load slowdown (ms)", 0, 3000, int(s["load_slowdown_ms"]))
    s["rate_limit_every"] = c5.number_input("Rate-limit every Nth request (0=off)", 0, 100, int(s.get("rate_limit_every",0)))
    s["tokens_per_minute"] = c6.number_input("Stand-in TPM budget", 100, 100000, int(s.get("tokens_per_minute",12000)))
    if st.button("Save stand-in settings"):
        cfg["standin"] = s
        cfg["groq_model"] = groq_model
        save_config(cfg)
        st.success("Stand-in settings saved.")

st.divider()
st.caption("Submission discipline: keep your API key private; preserve the raw CSV; do not report stand-in observations as real Groq measurements.")
