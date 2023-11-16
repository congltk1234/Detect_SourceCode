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


def guessLang(raw_text):
    '''
    Extract Code block using GuessLang
    '''
    response = {}
    count = 0

    for block in preprocess(raw_text):
        name = guess.language_name(block+'\n').lower()
        if name not in support_lang:
            name = 'Not SourceCode'
        else:
            response[f'SourceCode {count}'] = {'language':name, 'source':block}
            count+=1
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
    name = pipeline(text)[0]
    return name