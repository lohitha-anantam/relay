from pathlib import Path
import pandas as pd, threading
ROOT=Path(__file__).resolve().parents[1]
RESULTS_PATH=ROOT/"results.csv"
COLUMNS=["request_id","scenario","source","start_time","first_token_ms","total_ms","status","retries","tokens_in","tokens_out","retrieval_setting","answer_grade"]
_lock=threading.Lock()

def append_result(row):
    with _lock:
        df=pd.DataFrame([row],columns=COLUMNS)
        header=not RESULTS_PATH.exists()
        df.to_csv(RESULTS_PATH,mode="a",header=header,index=False)

def load_results():
    if not RESULTS_PATH.exists(): return pd.DataFrame(columns=COLUMNS)
    try: return pd.read_csv(RESULTS_PATH)
    except Exception: return pd.DataFrame(columns=COLUMNS)
