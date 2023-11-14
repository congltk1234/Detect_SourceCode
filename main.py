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

# @app.get("/guesslang")
# async def test_guesslang():
#     return {"message": "Guesslang Test"}

## Guesslang
@app.post("/guesslang")
async def language_name(item: Item):
    name = guess.language_name(item.text)
    print(name)
    return name