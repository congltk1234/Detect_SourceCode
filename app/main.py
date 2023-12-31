##
# @mainpage Detect Sourcecode Project
#
# @section description_main Description
# This application that allows users to identify programming languages using natural language processing techniques. This API is designed to be simple and user-friendly, with an intuitive interface and easy-to-understand documentation.
#
# @section libraries_main Libraries/Modules
# - @b FastAPI (https://fastapi.tiangolo.com/)
#   - FastAPI is a modern, high-performance Python web framework that enables developers to build APIs with ease. It offers a user-friendly interface and generates comprehensive documentation automatically, ensuring seamless integration and easy testing for your APIs.
# - @b guesslang (https://github.com/yoeo/guesslang)
# - @b CodeBERT (https://huggingface.co/huggingface/CodeBERTa-language-id)
#
#  @section libraries_main Endpoints
# - @b /
#   - This is the root endpoint of the API. By accessing this endpoint, users can be redirected to the API documentation.
# - @b /guesslang
#   - This endpoint is used to submit input text to the Guesslang model.
#   - The API will then return a JSON response containing the identified programming language.
# - @b /guesslang/extract
#   - This endpoint is used to Extracts source code and identified programming languages from the input text using Guesslang model.
#   - The API will then return a JSON response containing the sourcecode and its identified programming language.
# - @b /CodeBERT
#   - This endpoint is used to submit input text to the CodeBERT model.
#   - The API will then return a JSON response containing the identified programming language.
#
# @section author_doxygen_example Author(s)
# - Created by st_cong on 15/12/2023.
#
# Copyright (c) 2023 Brycen Vietnam Co., Ltd.  All rights reserved.


##
# @file main.py
#
# @section description_main Description
# This FastAPI application that allows users to identify programming languages using natural language processing techniques. This API is designed to be simple and user-friendly, with an intuitive interface and easy-to-understand documentation.
#
#  @section endpoints_main Endpoints
# - @b /
#   - This is the root endpoint of the API. By accessing this endpoint, users can be redirected to the API documentation.
# - @b /guesslang
#   - This endpoint is used to submit input text to the Guesslang model.
#   - The API will then return a JSON response containing the identified programming language.
# - @b /guesslang/extract
#   - This endpoint is used to Extracts source code and identified programming languages from the input text using Guesslang model.
#   - The API will then return a JSON response containing the sourcecode and its identified programming language.
# - @b /CodeBERT
#   - This endpoint is used to submit input text to the CodeBERT model.
#   - The API will then return a JSON response containing the identified programming language.
#
# Copyright (c) 2023 Brycen Vietnam Co., Ltd.  All rights reserved.


from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import time
from func.detect import *

from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class SimpleModel:
    nm: str = Form(...)

class Item(BaseModel):
    text: str

# Global Constants
## Initialize FastAPI application
app = FastAPI()
templates = Jinja2Templates(directory='templates')

# @app.get("/", include_in_schema=False)
# async def redirect():
#     return RedirectResponse("/docs")
@app.get('/', response_class=HTMLResponse, include_in_schema=False)
def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post("/form")
def form_post(form_data: SimpleModel = Depends()):
    return form_data
# https://stackoverflow.com/questions/60127234/how-to-use-a-pydantic-model-with-form-data-in-fastapi
# https://stackoverflow.com/questions/74504161/how-to-submit-selected-value-from-html-dropdown-list-to-fastapi-backend
# https://stackoverflow.com/questions/74318682/how-to-submit-html-form-input-value-using-fastapi-and-jinja2-templates


@app.post("/guesslang")
async def guesslang_detector(item: Item):
    """! Returns the identified programming language from the input text using guesslang.

    @param item   The input text to process.

    @return The identified programming language.
    """
    start_time = time.time()
    response = guessLang(item.text)
    print("Time took to process the request and return response is {} sec".format(time.time() - start_time))
    return {"name" : response}

@app.post("/guesslang/extract")
async def extract_and_detect_by_Guesslang(item: Item):
    """! Extracts source code and identified programming languages from the input text using guesslang.

    @param item   The input text to process.

    @return The extracted programming languages.
    """
    start_time = time.time()
    response = guessLang_extract(item.text)
    print("Time took to process the request and return response is {} sec".format(time.time() - start_time))

    return response


@app.post("/CodeBERT")
async def CodeBert_detector(item: Item):
    """! Returns the identified programming language from the input text using CodeBERT.

    @param item   The input text to process.

    @return The identified programming language.
    """

    start_time = time.time()
    response = codeBERT(item.text)
    print("Time took to process the request and return response is {} sec".format(time.time() - start_time))

    return {"name" : response}
