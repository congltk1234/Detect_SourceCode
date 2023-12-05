# Define Code Languages Scope:
support_lang = ['python', 'c', 'java', 'javascript', 'php', 'ruby', 'go', 'html', 'css']


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

def initialize_Guesslang_models():
    guess = Guess()
    return guess


def guessLang(block):
    '''
    Classify codeblock language using GuessLang
    '''
    block = str(block)
    if len(block) >0:
        guess = initialize_Guesslang_models()
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

from transformers import TextClassificationPipeline,RobertaTokenizer, RobertaForSequenceClassification

def initialize_CODEBERT_models():
    CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"
    tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
    model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)
    pipeline = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer
        )
    return pipeline


def codeBERT(text):
    '''
    Classify codeblock language using codeBERT
    '''
    pipeline = initialize_CODEBERT_models()
    name = pipeline(text)[0]
    return name