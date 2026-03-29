from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import summarize_article

app = FastAPI()

class Request(BaseModel):
    text: str
    prompt_version: str = "v1"

@app.post("/summarize")
def summarize(req: Request):
    return {"summary": summarize_article(req.text, req.prompt_version)}