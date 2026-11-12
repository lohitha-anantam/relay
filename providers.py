import requests, time, random, threading, json

class ProviderError(Exception):
    def __init__(self, message, status="error", retry_after=0, headers=None):
        super().__init__(message); self.status=status; self.retry_after=retry_after; self.headers=headers or {}

class GroqProvider:
    def __init__(self, api_key, model):
        self.api_key=api_key; self.model=model; self.lock=threading.Lock(); self.last_headers={}
    def complete(self, messages, stream=True):
        url="https://api.groq.com/openai/v1/chat/completions"
        payload={"model":self.model,"messages":messages,"temperature":0.1,"max_tokens":700,"stream":stream, "stream_options":{"include_usage":True} if stream else None}
        started=time.perf_counter()
        try:
            r=requests.post(url,headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"},json=payload,stream=stream,timeout=(15,90))
        except requests.RequestException as e:
            raise ProviderError(str(e),"timeout")
        self.last_headers=dict(r.headers)
        if r.status_code==429:
            body=r.text[:1000]
            raise ProviderError(f"Groq rate limit: {body}","rate_limited",float(r.headers.get("retry-after",0) or 0),dict(r.headers))
        if r.status_code>=400:
            raise ProviderError(f"Groq HTTP {r.status_code}: {r.text[:1000]}","error",headers=dict(r.headers))
        first=None; text=""; usage={}
        if stream:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                data=line[5:].strip()
                if data=="[DONE]": break
                try: obj=json.loads(data)
                except Exception: continue
                if obj.get("usage"): usage=obj["usage"]
                choices=obj.get("choices",[])
                if choices:
                    delta=choices[0].get("delta",{}).get("content") or ""
                    if delta:
                        if first is None: first=(time.perf_counter()-started)*1000
                        text+=delta
        else:
            obj=r.json(); text=obj["choices"][0]["message"]["content"]; usage=obj.get("usage",{})
            first=(time.perf_counter()-started)*1000
        self.last_usage=usage
        self.last_first_token=first if first is not None else (time.perf_counter()-started)*1000
        return text, self.last_first_token, usage, self.last_headers

class StandInProvider:
    _lock=threading.Lock(); _count=0; _inflight=0
    def __init__(self,base_latency_ms=900,jitter_ms=250,failure_rate=.03,rate_limit_every=0,load_slowdown_ms=120,tokens_per_minute=12000,stream_chunk_chars=18,daily_exhausted=False):
        self.base=base_latency_ms; self.jitter=jitter_ms; self.failure=failure_rate; self.rate_every=rate_limit_every
        self.slow=load_slowdown_ms; self.tpm=tokens_per_minute; self.chunk=stream_chunk_chars; self.daily_exhausted=daily_exhausted
    def complete(self,messages,stream=True):
        with self._lock:
            StandInProvider._count+=1; StandInProvider._inflight+=1
            count=StandInProvider._count; inflight=StandInProvider._inflight
        start=time.perf_counter()
        try:
            if self.daily_exhausted: raise ProviderError("Stand-in daily budget exhausted","rate_limited")
            if self.rate_every and count%self.rate_every==0: raise ProviderError("Stand-in per-minute limit reached","rate_limited",retry_after=2)
            if random.random()<self.failure: raise ProviderError("Injected stand-in failure","error")
            delay=max(30,self.base+random.uniform(-self.jitter,self.jitter)+(inflight-1)*self.slow)
            time.sleep(delay/1000)
            user=messages[-1]["content"] if messages else ""
            # Deterministic response: the actual pipeline still verifies citation support.
            if "JSON" in user or "json" in user.lower():
                answer='{"intent":"research_question","needs_retrieval":true}'
            elif "verify" in user.lower():
                answer="The answer uses the retrieved evidence and includes page citations. No unsupported claims were intentionally added."
            elif "plan" in user.lower():
                answer="Answer only from retrieved passages. If evidence is insufficient, refuse clearly."
            else:
                answer="The supplied passages describe the relevant method and its design choices. The retrieved text supports only the claims stated here; consult the cited paper pages for the precise details. [Source: retrieved passage]"
            first_delay=min(delay,delay*.65)
            time.sleep(max(0,first_delay/1000))
            first=(time.perf_counter()-start)*1000
            if stream:
                for _ in range(0,len(answer),self.chunk): time.sleep(.004)
            return answer,first,{"prompt_tokens":max(1,len(str(messages))//4),"completion_tokens":max(1,len(answer)//4)},{"x-standin-inflight":str(inflight),"x-standin-count":str(count)}
        finally:
            with self._lock: StandInProvider._inflight=max(0,StandInProvider._inflight-1)
