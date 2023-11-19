import re
from guesslang import Guess

guess = Guess()

# Define Code Languages Scope:
support_lang = ['python', 'c', 'java', 'javascript', 'php', 'ruby', 'go', 'html', 'css']


def preprocess(raw_text)->list:
    '''
    Split raw text into list of blocks
    '''
    return raw_text.split('\n\n')


def guessLang(block):
    '''
    Classify codeblock language using GuessLang
    '''
    name = guess.language_name(block+'\n').lower()
    return name


def guessLang_extract(raw_text):
    '''
    Extract Code block using GuessLang
    '''
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
    


from transformers import TextClassificationPipeline,RobertaTokenizer, RobertaForSequenceClassification
CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"
tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)
pipeline = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer
    )

def codeBERT(text):
    '''
    Classify codeblock language using codeBERT
    '''
    name = pipeline(text)[0]
    return name