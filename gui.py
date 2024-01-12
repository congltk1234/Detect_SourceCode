import tkinter as tk
import time
import torch

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

def guessLang_classify(block):
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


def guessLang(block):
    '''
    Classify codeblock language using GuessLang
    '''
    start_time = time.time()
    block = str(block)
    block = block.strip()
    if len(block) >0:
        language_probabilities = guess.language_name(block)
        language_name, _ = language_probabilities[0]
        return {"name" : language_name.lower(),
                'time' : time.time() - start_time}


def guessLang_extract(raw_text):
    '''
    Extract Code block using GuessLang
    '''
    response = {}
    count = 0
    list_block = preprocess(raw_text)
    source = ''
    for i in range(len(list_block)):
        name = guessLang_classify(list_block[i])
        try:
            not_code = name['not_code']
            if len(list_block[i])> 80:
                not_code += 0.2
            if name['code']> not_code:
                source+='\n' + list_block[i]
                response[f'Line {count}'] = {
                    'score':name, 'source':list_block[i]}
                count+=1
            else:
                source+='\n'
        except:
            continue
    print(response)
    if len(response)>0:
        return {'msg' : 'There are source code in input text. Cannot go to next step',
                'bool': False,
                'Inline source' : source}
    else:
        return {'msg' : 'No Sourcecode found',
                'bool': True,}

##########################################
############ CODEBERT ####################
##########################################

def codeBERT(CODE_TO_IDENTIFY):
    '''
    Classify codeblock language using codeBERT
    '''
    start_time = time.time()
    if len(CODE_TO_IDENTIFY) == 0:
        return {'msg': 'No SourceCode found'}
    
    inputs = tokenizer(CODE_TO_IDENTIFY, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    result = model.config.id2label[predicted_class_id]
    return {"name" : result,
            'time' : time.time() - start_time}
    
# This is a scrollable text widget
class ScrollText(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.text = tk.Text(self,selectbackground="gray", width=90, height=30)

        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)

        self.numberLines = TextLineNumbers(self, width=40)
        self.numberLines.attach(self.text)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.numberLines.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
        self.text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.text.bind("<Key>", self.onPressDelay)
        self.text.bind("<Button-1>", self.numberLines.redraw)
        self.scrollbar.bind("<Button-1>", self.onScrollPress)
        self.text.bind("<MouseWheel>", self.onPressDelay)

    def onScrollPress(self, *args):
        self.scrollbar.bind("<B1-Motion>", self.numberLines.redraw)

    def onScrollRelease(self, *args):
        self.scrollbar.unbind("<B1-Motion>", self.numberLines.redraw)

    def onPressDelay(self, *args):
        self.after(2, self.numberLines.redraw)

    def get(self, *args, **kwargs):
        return self.text.get(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self.text.insert(*args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.text.delete(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.text.index(*args, **kwargs)

    def redraw(self):
        self.numberLines.redraw()


'''THIS CODE IS CREDIT OF Bryan Oakley (With minor visual modifications on my side): 
https://stackoverflow.com/questions/16369470/tkinter-adding-line-number-to-text-widget'''


class TextLineNumbers(tk.Canvas):
    def __init__(self, *args, **kwargs):
        tk.Canvas.__init__(self, *args, **kwargs, highlightthickness=0)
        self.textwidget = None

    def attach(self, text_widget):
        self.textwidget = text_widget

    def redraw(self, *args):
        '''redraw line numbers'''
        self.delete("all")

        i = self.textwidget.index("@0,0")
        while True :
            dline= self.textwidget.dlineinfo(i)
            if dline is None: break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(2, y, anchor="nw", text=linenum, fill="blue")
            i = self.textwidget.index("%s+1line" % i)


def var_states():
    # print(f"Guesslang: {str(guesslang_check.get())},\nCodeBERT: {str(codebert_check.get())},\nMode: {var.get()}")  
    print(f"Mode: {var.get()}")  
    input_text = scroll.get("1.0","end-1c")
    if var.get() == 'Detector':
        result_codebert = codeBERT(input_text)
        result_guesslang = guessLang(input_text)
        L.configure(text=f'Guesslang:\n    name: {result_guesslang["name"]}\n    time: {result_guesslang["time"]}\n\nCodeBERT:\n    name: {result_codebert["name"]}\n    time: {result_codebert["time"]}', foreground="green")
    else:
        extractor = guessLang_extract(input_text)
        if not extractor["bool"]:
            L.configure(text=f'{extractor["msg"]}\n\n{extractor["Inline source"]}', foreground="red")
        else:
            L.configure(text=f'{extractor["msg"]}', foreground="green")


import numpy as np
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
        code_score = code/(code+notcode) 
        not_score = notcode/(code+notcode)
        if code_score > not_score:
            return 1.
        else:
            return 0.


def confusion_matrix(y_true, y_pred):
    # Ensure input arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate true positives, false positives, false negatives, and true negatives
    tp = np.sum((y_true == 1.) & (y_pred == 1.))
    fp = np.sum((y_true == 0.) & (y_pred == 1.))
    fn = np.sum((y_true == 1.) & (y_pred == 0.))
    tn = np.sum((y_true == 0.) & (y_pred == 0.))

    # Create the confusion matrix
    matrix = np.array([[tp, fp],
                       [fn, tn]])

    print('confusion matrix\n',matrix)

def f1_score(y_true, y_pred):
    # Ensure input arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate true positives, false positives, and false negatives
    tp = np.sum((y_true == 1.) & (y_pred == 1.))
    fp = np.sum((y_true == 0.) & (y_pred == 1.))
    fn = np.sum((y_true == 1.) & (y_pred == 0.))

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print('precision',precision)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print('recall',recall)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1

def predict(file_name):
    source_code = open(file_name, 'r')
    source_code = source_code.readlines()
    for i in source_code:
        if guessLang_evaluate(i)==1:
            print('predict: This file contains Source Code')
            L.configure(text=f'This file contains Source Code', foreground="red")
            return False
    





def evaluate():
    normal_text = open('evaluate/normal_text.txt', 'r')
    normal_text = normal_text.readlines()
    source_code = open('evaluate/source_code.txt', 'r')
    source_code = source_code.readlines()
    X = normal_text + source_code

    predict = [guessLang_evaluate(i) for i in X]
    predict = np.array(predict)        

    y_label = np.concatenate((np.zeros(100), np.ones(100)), axis=None)

    print('y_label:', y_label)
    print('predict:', predict)
    confusion_matrix(y_label, predict)
    accuracy = (y_label == predict).mean()
    print('accuracy', accuracy)
    score = f1_score(y_label,predict)
    print('f1_score', score)



import tkinter.filedialog
# file explorer window
def browseFiles():
    filename = tkinter.filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
      
    # Change label contents
    print(filename)
    predict(filename)


if __name__ == '__main__':
    # try:
    #     evaluate()
    # except:
    #     pass
    root = tk.Tk()
    scroll = ScrollText(root)
    # scroll.insert(tk.END, "HEY" + 20*'\n')
    scroll.pack(expand=True, fill='both', side='left')
    scroll.text.focus()
    root.after(200, scroll.redraw())
    

    label = tk.Label(root, text = "Module Options:")
    label.pack(side=tk.TOP)
    var = tk.StringVar(value=' ')
    R1 = tk.Radiobutton(root, text="Detector", variable=var, value='Detector')
    R1.pack( anchor = tk.W )
    R2 = tk.Radiobutton(root, text="Extractor", variable=var, value='Extractor')
    R2.pack( anchor = tk.W)


    # model_select = tk.Label(root, text="Model selection:").pack( anchor = tk.W)
    # guesslang_check = tk.IntVar()
    # codebert_check = tk.IntVar()

    # tk.Checkbutton(root, text="Guesslang", variable=guesslang_check).pack( anchor = tk.W)
    # tk.Checkbutton(root, text="CodeBERT", variable=codebert_check).pack( anchor = tk.W)
    tk.Button(root, text='Predict', command=var_states).pack( anchor = tk.W)
    # Create a File Explorer label
    tk.Button(root, text = "Browse Files", command = browseFiles).pack( anchor = tk.W) 
    tk.Label(root,text="Result").pack(anchor=tk.W)
    L = tk.Label(root, text="Hello!", relief="sunken", bg = "light yellow", justify="left")
    L.pack()


    root.mainloop()