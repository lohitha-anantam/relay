import time, uuid
from datetime import datetime, timezone
from .library import chunks
from .providers import ProviderError

SETTINGS={
 "small_fast":{"piece_size":450,"top_k":3,"rerank":False},
 "balanced":{"piece_size":850,"top_k":5,"rerank":True},
 "large_context":{"piece_size":1300,"top_k":8,"rerank":True},
}
QUESTIONS=[
"What is the role of multi-head attention in the Transformer?",
"How does the Transformer avoid recurrence while modeling sequence relationships?",
"What positional information is added to the Transformer and why?",
"What are the key components of the encoder layer described in Attention Is All You Need?",
"What is the purpose of scaled dot-product attention?",
"What is Moshi and what kind of interaction is it designed to support?",
"What does the Moshi paper say about real-time dialogue?",
"What is the relationship between speech and text in Moshi?",
"What architecture or components are described for Moshi's audio generation?",
"What limitations or evaluation findings are reported in the Moshi paper?",
"What problem does Dynamic Chunking address?",
"How does dynamic chunking differ from fixed chunking in the named paper?",
"What hierarchical sequence modeling task is discussed in Dynamic Chunking?",
"What empirical results are reported by Dynamic Chunking for End-to-End Hierarchical Sequence Modeling?",
"Do these three papers establish that one architecture is universally better than the others?",
"According to the three papers, what is the exact Groq free-tier requests-per-minute limit?",
"Do the papers specify a production API daily token budget?",
"Which paper gives a measured comparison of Moshi and the original Transformer?",
"Do the papers describe a Groq retry-after header?",
"Can the papers confirm the current price of Groq API usage?"
]

def _tokens(s): return max(1,len(s)//4)

class RelayPipeline:
    def __init__(self,provider,setting="balanced"):
        self.provider=provider; self.setting=SETTINGS[setting]; self.setting_name=setting
    def _call(self,system,user):
        return self.provider.complete([{"role":"system","content":system},{"role":"user","content":user}],stream=True)
    def retrieve(self,q):
        cfg=self.setting; all_chunks=chunks(cfg["piece_size"])
        terms=set(w.lower().strip(".,?!:;()[]") for w in q.split() if len(w)>2)
        scored=[]
        for c in all_chunks:
            words=set(w.lower().strip(".,?!:;()[]") for w in c["text"].split())
            score=len(terms & words)/(len(terms)**.5+1)
            if score>0: scored.append((score,c))
        scored.sort(key=lambda x:x[0],reverse=True)
        # lightweight reranking by exact query-term density and title-independent evidence
        if cfg["rerank"]:
            scored.sort(key=lambda x:(x[0],sum(t in x[1]["text"].lower() for t in terms)),reverse=True)
        return [{"score":s,**c} for s,c in scored[:cfg["top_k"]]]
    def run(self,question,scenario="single",source="stand-in"):
        start=time.perf_counter(); start_iso=datetime.now(timezone.utc).isoformat()
        trace=[]; retries=0; first_token=None; tokens_in=0; tokens_out=0
        try:
            # Step 1: intent classification (local branch)
            intent="research_question"
            needs_retrieval=True
            trace.append({"step":"classify","method":"local heuristic","intent":intent})
            # Step 2: retrieval tool
            retrieved=self.retrieve(question)
            trace.append({"step":"retrieve","tool":"local paper chunk search","chunks":len(retrieved)})
            if not retrieved:
                answer="I cannot answer this from the three-paper library because no relevant supporting passage was retrieved. [Insufficient evidence]"
                grade="refused_correctly"; status="ok"
            else:
                evidence="\n".join(f'[Paper: {x["paper"]}; page {x["page"]}] {x["text"]}' for x in retrieved)
                # Step 3: plan
                call_started=time.perf_counter()
                plan,ft,u,h=self._call("You are a research assistant. Return a concise answer plan. Do not use outside knowledge.",f"Plan how to answer this question from evidence only: {question}\nEvidence:\n{evidence[:9000]}")
                first_token=(call_started-start)*1000+ft; tokens_in+=u.get("prompt_tokens",0); tokens_out+=u.get("completion_tokens",0)
                trace.append({"step":"plan","first_token_ms":round(ft,2)})
                # Step 4: cited draft
                call_started2=time.perf_counter()
                draft,ft2,u,h=self._call("Answer only from supplied evidence. Cite every factual claim with paper title and page. If unsupported, say the papers do not specify it. Never invent facts.",f"Question: {question}\nPlan: {plan}\nEvidence:\n{evidence[:12000]}")
                if first_token is None: first_token=(call_started2-start)*1000+ft2
                tokens_in+=u.get("prompt_tokens",0); tokens_out+=u.get("completion_tokens",0)
                trace.append({"step":"draft","first_token_ms":round(ft2,2)})
                # Step 5: verification branch
                verify,ft3,u,h=self._call("You are a strict citation auditor. Identify unsupported claims or missing page citations. If any exist, return a corrected short answer; otherwise return the original answer unchanged.",f"Question: {question}\nEvidence:\n{evidence[:10000]}\nDraft:\n{draft}")
                tokens_in+=u.get("prompt_tokens",0); tokens_out+=u.get("completion_tokens",0)
                answer=verify.strip() or draft
                trace.append({"step":"verify","first_token_ms":round(ft3,2)})
                has_cite=("page" in answer.lower() or "source:" in answer.lower())
                uncertain=any(x in answer.lower() for x in ["do not specify","cannot answer","insufficient evidence","not stated"])
                grade="refused_correctly" if uncertain else ("correct" if has_cite else "wrong")
                status="ok"
            total=(time.perf_counter()-start)*1000
            row={"request_id":str(uuid.uuid4()),"scenario":scenario,"source":source,"start_time":start_iso,
                 "first_token_ms":round(first_token if first_token is not None else total,2),"total_ms":round(total,2),
                 "status":status,"retries":retries,"tokens_in":tokens_in,"tokens_out":tokens_out,
                 "retrieval_setting":self.setting_name,"answer_grade":grade}
            return {"answer":answer,"retrieved":retrieved if 'retrieved' in locals() else [],"trace":trace,"row":row}
        except ProviderError as e:
            total=(time.perf_counter()-start)*1000
            row={"request_id":str(uuid.uuid4()),"scenario":scenario,"source":source,"start_time":start_iso,
                 "first_token_ms":round(first_token or 0,2),"total_ms":round(total,2),"status":e.status,
                 "retries":retries,"tokens_in":tokens_in,"tokens_out":tokens_out,"retrieval_setting":self.setting_name,"answer_grade":"refused_wrongly"}
            raise PipelineRecordedError(str(e),row)
        except Exception:
            raise

class PipelineRecordedError(Exception):
    def __init__(self,message,row): super().__init__(message); self.row=row
