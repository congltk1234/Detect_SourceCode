import time
root_time = time.time() 
import torch
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse
from dataclasses import dataclass
from fastapi.templating import Jinja2Templates

@dataclass
class SimpleModel:
    input_text: str = Form(...)


def preprocess(raw_text):
    '''
    Split raw text into list of blocks
    '''
    raw_text = str(raw_text)
    list_block = raw_text.split('\n')
    return list_block


##########################################
############ GUESSLANG ###################
##########################################
from guesslang import Guess
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def load_model():
    start_time = time.time()
    tokenizer = RobertaTokenizer.from_pretrained("saved_model/", local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained("saved_model/",  local_files_only=True)
    guess = Guess()
    print("Success to load model in {} sec".format(time.time() - start_time))
    return guess,tokenizer,model

guess,tokenizer,model = load_model()

# Define Code Languages Scope:
# support_lang = ['python', 'c', 'java', 'javascript', 'php', 'ruby', 'go', 'html', 'css']
not_code = ['tex',  'ini', 'csv' , 'batchfile', 'markdown', 'json', 'prolog' , 'yaml', 'sql', 'powershell', 'dockerfile', 'shell', 'makefile', 'fortran']
def guessLang(block):
    '''
    Classify codeblock language using GuessLang
    '''
    block = str(block)
    block = block.strip()
    if len(block) >0:
        name = guess.language_name(block)
        code = sum(value for _, value in name[:28] if _.lower() not in  not_code)
        notcode = sum(value for _, value in name if _.lower() in not_code)
        return {'code': code/(code+notcode), 'not_code':notcode/(code+notcode)}
    


def guessLang_extract(raw_text):
    '''
    Extract Code block using GuessLang
    '''
    response = {}
    count = 0
    list_block = preprocess(raw_text)
    source = ''
    for i in range(len(list_block)):
        name = guessLang(list_block[i])
        try:
            not_code = name['not_code'] + 0.2 if len(list_block[i]) > 80 else 0
            if name['code']> not_code:
                source+='\n' + list_block[i]
                response[f'Line {count}'] = {
                    'score':name, 'source':list_block[i]}
                count+=1
            else:
                source+='\n'
        except:
            continue
    return {'msg' : 'There are source code in input text. Cannot go to next step',
            'Inline source' : source} 

def guessLang_evaluate(block):
    '''
    Classify codeblock language using GuessLang
    '''
    block = str(block)
    block = block.strip()
    if len(block) >0:
        name = guess.language_name(block)
        code = sum(value for _, value in name[:28] if _.lower() not in  not_code)
        notcode = sum(value for _, value in name if _.lower() in not_code)
        code, notcode = code/(code+notcode) , notcode/(code+notcode)
        if code> not_code:
            return 1
        else:
            return 0

##########################################
############ CODEBERT ####################
##########################################



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
templates = Jinja2Templates('templates')
print("Success to init API : {} sec".format(time.time() - root_time))


@app.get('/', response_class=HTMLResponse, include_in_schema=False)
def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


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
async def guesslang_detector(form_data: SimpleModel = Depends()):
    """! Returns the identified programming language from the input text using guesslang.

    @param item   The input text to process.

    @return The identified programming language.
    """
    start_time = time.time()
    response = guessLang(form_data.input_text.replace("\r", ""))
    print("Time took to process the request and return response is {} sec".format(time.time() - start_time))
    
    return {"name" : response,
            'time' : time.time() - start_time}



@app.post("/guesslang/extract",tags=["guesslang"])
async def extract_and_detect_by_Guesslang(form_data: SimpleModel = Depends()):
    """! Extracts source code and identified programming languages from the input text using guesslang.

    @param item   The input text to process.

    @return The extracted programming languages.
    """
    start_time = time.time()
    response = guessLang_extract(form_data.input_text.replace("\r", ""))
    print("Time took to process the request and return response is {} sec".format(time.time() - start_time))
    if len(response) == 0:
        return {'msg': 'No SourceCode found',
                'time' : time.time() - start_time}
    else:
        return {'response' :response,
                'time' : time.time() - start_time}
    

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
async def CodeBert_detector(form_data: SimpleModel = Depends()):
    """! Returns the identified programming language from the input text using CodeBERT.

    @param item   The input text to process.

    @return The identified programming language.
    """

    start_time = time.time()
    response = codeBERT(form_data.input_text.replace("\r", ""))
    print("Time took to process the request and return response is {} sec".format(time.time() - start_time))

    return {"name" : response,
            'time' : time.time() - start_time}


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)