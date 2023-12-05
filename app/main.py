from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.func.detect import *

from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

## Guesslang
@app.post("/guesslang")
async def guesslangAPI(item: Item):
    response = guessLang(item.text)
    return response

@app.post("/guesslang/extract")
async def guesslangExtract(item: Item):
    response = guessLang_extract(item.text)
    return response


## CodeBERT
@app.post("/CodeBERT")
async def codebertTest(item: Item):
    return codeBERT(item.text)
