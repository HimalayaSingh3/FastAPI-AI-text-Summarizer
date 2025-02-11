from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="AI Text Summarizer", description="Summarize long texts using AI", version="1.0")

summarizer = pipeline("summarization")

class TextRequest(BaseModel):
    text: str

@app.post("/summarize/")
def summarize_text(request: TextRequest):
    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="Text is too short to summarize")
    
    summary = summarizer(request.text, max_length=150, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

@app.get("/")
def root():
    return {"message": "Welcome to the AI Text Summarizer! Use /docs to test the API."}
