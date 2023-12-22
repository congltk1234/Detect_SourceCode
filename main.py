import json
import yaml
import torch
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Body
from fastapi.responses import RedirectResponse
from typing import Annotated

class Item(BaseModel):
    text: str



def preprocess(raw_text)->list:
    '''
    Split raw text into list of blocks
    '''
    raw_text = str(raw_text)
    return raw_text.split('\n\n')


##########################################
############ GUESSLANG ###################
##########################################
from guesslang import Guess

guess = Guess()

# Define Code Languages Scope:
support_lang = ['python', 'c', 'java', 'javascript', 'php', 'ruby', 'go', 'html', 'css']

def guessLang(block):
    '''
    Classify codeblock language using GuessLang
    '''
    block = str(block)
    if len(block) >0:
        name = guess.language_name(block+'\n')
        return name.lower()
    else:
        return None


def guessLang_extract(raw_text):
    '''
    Extract Code block using GuessLang
    '''
    if raw_text == None:
        return {'msg': 'No SourceCode found'}
    
    response = {}
    count = 0
    for block in preprocess(raw_text):
        name = guessLang(block)
        if name not in support_lang:
            name = 'Not SourceCode'
        else:
            response[f'SourceCode {count}'] = {'language':name, 'source':block}
            count+=1

    if len(response) == 0:
        return {'msg': 'No SourceCode found'}
    else:
        return response

##########################################
############ CODEBERT ####################
##########################################

from transformers import RobertaTokenizer, RobertaForSequenceClassification

CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"

tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)


def codeBERT(CODE_TO_IDENTIFY):
    '''
    Classify codeblock language using codeBERT
    '''
    if len(CODE_TO_IDENTIFY) == 0:
        return {'msg': 'No SourceCode found'}
    
    inputs = tokenizer(CODE_TO_IDENTIFY, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    result = model.config.id2label[predicted_class_id]
    return result


####################################
############ API ###################
####################################

tags_metadata = [
    {
        "name": "guesslang",
        "description": "Detect source code with **guesslang**.",
        "externalDocs": {
            "description": "guesslang",
            "url": "https://github.com/yoeo/guesslang",
        },
    },
    {
        "name": "codebert",
        "description": "Detect source code with **CodeBERT**.",
        "externalDocs": {
            "description": "CodeBERT",
            "url": "https://huggingface.co/huggingface/CodeBERTa-language-id",
        },
    },
]

app = FastAPI(title="Brycen GPT Filter Application",
    openapi_tags=tags_metadata)

@app.on_event("startup")
def save_openapi_spec():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)
    with open("openapi.yaml", "w") as file:
        yaml.dump(openapi_data, file, sort_keys=False)


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


sample = '''for i in range(n):
    print("Hello, World!")'''

## Guesslang
@app.post("/guesslang",tags=["guesslang"],     
    responses={
        200: {
            "description": "Return the program language name detected by guesslang.",
            "content": {
                "application/json": {
                    "example": {"name": "python"}
                }
            }
        }
    },)
async def guesslang_detector(item: Annotated[
        Item,
        Body(
            examples=[
                {
                    "text": sample,
                }
            ],
        ),
    ],):
    response = guessLang(item.text)
    return {"name" : response}




large_txt = '''This is a sample for test:

#include <stdio.h>
int main() {
   printf("Hello, World!");
}

Another for test

for i in range(n):\n    print("Hello, World!")'''



@app.post("/guesslang/extract",tags=["guesslang"],     
    responses={
        200: {
            "description": "Return the codeblock and program language name detected by guesslang.",
            "content": {
                "application/json": {
                    "example": 
                {
  "SourceCode 0": {
    "language": "c",
    "source": "#include <stdio.h>\nint main() {\n   printf(\"Hello, World!\");\n}"
  },
  "SourceCode 1": {
    "language": "python",
    "source": "for i in range(n):\n    print(\"Hello, World!\")"
  }
}
                }
            }
        }
    },)
async def extract_and_detect_by_Guesslang(item: Annotated[
        Item,
        Body(
            examples=[
                {
                    "text": large_txt,
                }
            ],
        ),
    ]):
    response = guessLang_extract(item.text)
    return response

# CodeBERT
@app.post("/CodeBERT",tags=["codebert"],
    responses={
        200: {
            "description": "Return the program language name detected by CodeBERT.",
            "content": {
                "application/json": {
                    "example": {"name": "python"}
                }
            }
        }
    })
async def CodeBert_detector(item: Annotated[
        Item,
        Body(
            description = 'Helllo',
            examples=[
                {
                    "text": sample,
                }
            ],
        ),
    ]):
    return {"name" : codeBERT(item.text)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)