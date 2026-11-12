from pathlib import Path
import requests, re, json
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "data" / "papers"
PAPERS = [
    ("1706.03762v7", "Attention Is All You Need", "https://arxiv.org/pdf/1706.03762v7"),
    ("2410.00037v1", "Moshi: a speech-text foundation model for real-time dialogue", "https://arxiv.org/pdf/2410.00037v1"),
    ("2507.07955v2", "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", "https://arxiv.org/pdf/2507.07955v2"),
]

def ensure_library():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    for pid, title, url in PAPERS:
        path = PAPERS_DIR / f"{pid}.pdf"
        if not path.exists():
            r = requests.get(url, timeout=90, headers={"User-Agent":"TheRelayAssignment/1.0"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"): raise RuntimeError(f"arXiv did not return a PDF for {pid}")
            path.write_bytes(r.content)
    return True

def library_status():
    return {pid: ("downloaded" if (PAPERS_DIR/f"{pid}.pdf").exists() else "not downloaded") for pid,_,_ in PAPERS}

def load_pages():
    ensure_library()
    pages=[]
    for pid,title,_ in PAPERS:
        reader=PdfReader(str(PAPERS_DIR/f"{pid}.pdf"))
        for ix,p in enumerate(reader.pages,1):
            text=p.extract_text() or ""
            text=re.sub(r"\s+"," ",text).strip()
            if text:
                pages.append({"paper":title,"paper_id":pid,"page":ix,"text":text})
    return pages

def chunks(piece_size=900):
    result=[]
    for page in load_pages():
        words=page["text"].split()
        step=max(100,int(piece_size*0.8))
        for start in range(0,len(words),step):
            txt=" ".join(words[start:start+piece_size])
            if txt:
                result.append({**page,"text":txt})
    return result
