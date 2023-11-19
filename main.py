from fastapi import FastAPI
import model
from func import detect

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/guesslang")
async def test_guesslang():
    return {"message": "Guesslang Test"}

## Guesslang
@app.post("/guesslang")
async def guesslangTest(item: model.Item):
    response = detect.guessLang(item.text)
    return response

@app.post("/guesslang/extract")
async def guesslangExtract(item: model.Item):
    response = detect.guessLang_extract(item.text)
    return response


## CodeBERT
@app.post("/CodeBERT")
async def codebertTest(item: model.Item):
    return detect.codeBERT(item.text)
