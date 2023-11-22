from fastapi import FastAPI
import app.detect

from pydantic import BaseModel
from typing import Union

class Item(BaseModel):
    text: str

app = FastAPI()


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

@app.get("/guesslang")
async def test_guesslang():
    return {"message": "Guesslang Test"}

## Guesslang
@app.post("/guesslang")
async def guesslangTest(item: Item):
    response = detect.guessLang(item.text)
    return response

@app.post("/guesslang/extract")
async def guesslangExtract(item: Item):
    response = detect.guessLang_extract(item.text)
    return response


## CodeBERT
@app.post("/CodeBERT")
async def codebertTest(item: Item):
    return detect.codeBERT(item.text)
