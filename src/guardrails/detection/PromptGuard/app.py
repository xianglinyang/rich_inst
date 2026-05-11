from typing import List, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Available models:
# Llama Prompt Guard 2 86M	81.2%
# Llama Prompt Guard 2 22M	78.4%

MODEL_ID_22M = "meta-llama/Llama-Prompt-Guard-2-22M"
MODEL_ID_86M = "meta-llama/Llama-Prompt-Guard-2-86M"
Label = Literal["benign", "prompt_injection"] # benign, malicious
RawLabel = Literal["LABEL_0", "LABEL_1"] # benign, malicious

# -------- Load PromptGuard models --------
DEVICE = 0 if torch.cuda.is_available() else -1

tokenizer_22m = AutoTokenizer.from_pretrained(MODEL_ID_22M)
model_22m = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID_22M,
    trust_remote_code=True,
)
tokenizer_86m = AutoTokenizer.from_pretrained(MODEL_ID_86M)
model_86m = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID_86M,
    trust_remote_code=True,
)

# We want scores we can threshold: pipeline returns label + score
clf_22m = pipeline(
    "text-classification",
    model=model_22m,
    tokenizer=tokenizer_22m,
    truncation=True,
    max_length=512,
    device=DEVICE,
)

clf_86m = pipeline(
    "text-classification",
    model=model_86m,
    tokenizer=tokenizer_86m,
    truncation=True,
    max_length=512,
    device=DEVICE,
)

# -------- API schema --------
app_22m = FastAPI(title="PromptGuard 22M Multi-class API", version="1.0")
app_86m = FastAPI(title="PromptGuard 86M Multi-class API", version="1.0")

class DetectRequest(BaseModel):
    texts: List[str]

class DetectItem(BaseModel):
    label: Label
    confidence: float

class DetectResponse(BaseModel):
    model: str
    results: List[DetectItem]

# -------- PromptGuard 22M API --------

@app_22m.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID_22M, "device": "cuda" if DEVICE == 0 else "cpu"}

@app_22m.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest):

    if not req.texts or len(req.texts) > 128:
        raise HTTPException(status_code=400, detail="texts must be 1..128 items")
    texts = [text[:20000] for text in req.texts]

    with torch.no_grad():
        out = clf_22m(texts)

    results: List[DetectItem] = []
    for o in out:
        raw_label = o['label']
        conf = float(o["score"])
        print(raw_label, conf)

        if raw_label == "LABEL_0":
            results.append(DetectItem(
                label="benign",
                confidence=conf,
            ))
            continue

        results.append(DetectItem(
            label="prompt_injection",
            confidence=conf,
        ))

    return DetectResponse(model=MODEL_ID_22M, results=results)


# -------- PromptGuard 86M API --------

@app_86m.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID_86M, "device": "cuda" if DEVICE == 0 else "cpu"}

@app_86m.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest):

    if not req.texts or len(req.texts) > 128:
        raise HTTPException(status_code=400, detail="texts must be 1..128 items")
    texts = [text[:20000] for text in req.texts]

    with torch.no_grad():
        out = clf_86m(texts)

    results_86m: List[DetectItem] = []
    for o in out:
        raw_label = o['label']
        conf = float(o["score"])
        print(raw_label, conf)
        if raw_label == "LABEL_0":
            results_86m.append(DetectItem(
                label="benign",
                confidence=conf,
            ))
            continue

        results_86m.append(DetectItem(
            label="prompt_injection",
            confidence=conf,
        ))

    return DetectResponse(model=MODEL_ID_86M, version="86m", results=results_86m)

# uvicorn app:app_22m --host 127.0.0.1 --port 12392
# uvicorn app:app_86m --host 127.0.0.1 --port 12393