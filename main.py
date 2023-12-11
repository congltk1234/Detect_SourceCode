import torch
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
from pydantic import BaseModel

from transformers import TextClassificationPipeline, RobertaTokenizer, RobertaForSequenceClassification

def initialize_CODEBERT_models():
    CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"
    tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
    model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)
    pipeline = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer
        )
    return pipeline

pipeline = initialize_CODEBERT_models()

# def codeBERT(text):
#     '''
#     Classify codeblock language using codeBERT
#     '''
#     if len(text) == 0:
#         return {'msg': 'No SourceCode found'}
    
#     name = pipeline(text)[0]
#     return name


class Item(BaseModel):
    text: str

app = FastAPI()


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

# CodeBERT
# @app.post("/CodeBERT")
# async def codebertTest(item: Item):
#     return codeBERT(item.text)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)