from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from guesslang import Guess

app = FastAPI()
guess = Guess()


class Item(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/guesslang")
async def test_guesslang():
    return {"message": "Guesslang Test"}

## Guesslang
@app.post("/guesslang")
async def guesslangTest(item: Item):
    name = guess.language_name(item.text)
    print(name)
    return name


from transformers import TextClassificationPipeline,RobertaTokenizer, RobertaForSequenceClassification
## CodeBERT
@app.post("/CodeBERT")
async def codebertTest(item: Item):
    CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"
    tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
    model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)

    pipeline = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer
    )
    name = pipeline(item.text)[0]
    print(name)
    return name
