##
# @file detect.py
#
# @section description_detect Description
# File detect.py uses the Guesslang library and CodeBERT model (from Huggingface) to detect the programming language of a given text input. It preprocesses the raw text by breaking it into list of text blocks, and then applies the model to identify the language of each block.
# 
# The supported languages include Python, C, Java, JavaScript, PHP, Ruby, Go, HTML, and CSS.
#
#
# Copyright (c) 2023 Brycen Vietnam Co., Ltd.  All rights reserved.



## Define Code Languages Support
support_lang = ['python', 'c', 'java', 'javascript', 'php', 'ruby', 'go', 'html', 'css']


def preprocess(raw_text)->list:
    """! Preprocesses the raw text by breaking it into list of text blocks.
    
    @param raw_text   The raw text input
    #return   The preprocessed list of text blocks
    """
    raw_text = str(raw_text)
    return raw_text.split('\n\n')


##########################################
############ GUESSLANG ###################
##########################################
from guesslang import Guess

def initialize_Guesslang_models()->Guess:
    """! Initialize Guesslang model.
    
    #return   Guesslang model
    """
    guess = Guess()
    return guess


def guessLang(block)->str:
    """! Identify programming language of the block using GuessLang model.
    
    @param block   The text input
    #return   The programming language name
    """
    block = str(block)
    if len(block) >0:
        guess = initialize_Guesslang_models()
        name = guess.language_name(block+'\n')
        return name.lower()
    else:
        return None


def guessLang_extract(raw_text)->dict:
    """! Extracts source code and identified programming languages from the input text using guesslang.
    
    @param block   The text input
    #return   The programming language name
    """
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

from transformers import TextClassificationPipeline, RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModel

def initialize_CODEBERT_models():
    """! Initialize CodeBERT model.
    
    #return   CodeBERT model
    """

    CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"
    # tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
    # model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)
    
    # tokenizer.save_pretrained('saved_model/')
    # model.save_pretrained('saved_model/')

    tokenizer = RobertaTokenizer.from_pretrained("saved_model/", local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained("saved_model/",  local_files_only=True)
   
    pipeline = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer
        )
    return pipeline


def codeBERT(text)->str:
    """! Identify programming language of the block using CodeBERT model
    
    @param text   The text input
    #return   The programming language name
    """
    if len(text) == 0:
        return {'msg': 'No SourceCode found'}
    pipeline = initialize_CODEBERT_models()
    name = pipeline(text)[0]
    return name