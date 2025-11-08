from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI()

# Models
class SummarizeRequest(BaseModel):
    text: str
    max_sentences: int = 3

class InvoiceRequest(BaseModel):
    text: str

# Root
@app.get("/")
def root():
    return {"message": "AI Business Tool is running!"}

# Sumarize
def _split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    text = req.text.strip()
    if not text:
        return {"summary": ""}

    sentences = _split_sentences(text)

    # tiny frequency-based scorer
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    stop = set(("a an the and or but if then else for with of on in to from by we our you your i me my they them "
                "their is are was were be being been it its this that these those as at not no yes do done did can could "
                "would should may might just very really more most less least").split())
    freqs = {}
    for w in words:
        if w not in stop:
            freqs[w] = freqs.get(w, 0) + 1

    def score(s: str) -> float:
        tokens = re.findall(r"[a-zA-Z0-9']+", s.lower())
        return sum(freqs.get(t, 0) for t in tokens) / (len(s) + 1)

    ranked = sorted(((score(s), i, s) for i, s in enumerate(sentences)), reverse=True)
    top = sorted(ranked[: req.max_sentences], key=lambda x: x[1])
    summary = " ".join(s for _, __, s in top)
    return {"summary": summary}

# Extract invoice
INVOICE_RE = re.compile(r"(?:invoice|inv)[-\s#:]*([A-Za-z0-9-]{3,})", re.I)
DATE_RE    = re.compile(r"(\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b)")
TOTAL_RE   = re.compile(r"(?:total|amount due|balance)[:\s]*\$?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+\.[0-9]{2})", re.I)
VENDOR_RE  = re.compile(r"(?:from|vendor|supplier)[:\s]*([A-Za-z][A-Za-z0-9 &.,'-]{2,})", re.I)

def _clean_amount(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except Exception:
        return 0.0

@app.post("/extract-invoice")
def extract_invoice(req: InvoiceRequest):
    text = req.text or ""
    inv  = INVOICE_RE.search(text)
    dt   = DATE_RE.search(text)
    tot  = TOTAL_RE.search(text)
    ven  = VENDOR_RE.search(text)

    return {
        "invoice_number": inv.group(1) if inv else None,
        "date": dt.group(1) if dt else None,
        "total": _clean_amount(tot.group(1)) if tot else None,
        "vendor": ven.group(1).strip() if ven else None,
    }


# Lead score
class LeadScoreRequest(BaseModel):
    company: str
    notes: str

POSITIVE = {
    "enterprise": 15,
    "pilot": 10,
    "budget": 10,
    "buying": 10,
    "approved": 8,
    "timeline": 5,
    "high priority": 8,
    "contract": 10,
    "po": 8,
}
NEGATIVE = {
    "research": -5,
    "just looking": -8,
    "no budget": -12,
    "next year": -6,
    "not a priority": -8,
    "stall": -6,
}

@app.post("/lead-score")
def lead_score(req: LeadScoreRequest):
    notes = (req.notes or "").lower()
    score = 0
    reasons: list[str] = []

    for k, w in POSITIVE.items():
        if k in notes:
            score += w
            reasons.append(f"Found positive signal: '{k}' (+{w})")
    for k, w in NEGATIVE.items():
        if k in notes:
            score += w
            reasons.append(f"Found negative signal: '{k}' ({w})")

    # tiny heuristic: longer company names likely established orgs
    if req.company and len(req.company) > 10:
        score += 2
        reasons.append("Company name length suggests established org (+2)")

    return {"score": score, "reasons": reasons}
