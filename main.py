from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from detect import guessLang
from detect import guessLang_extract
from detect import codeBERT

from pydantic import BaseModel

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
