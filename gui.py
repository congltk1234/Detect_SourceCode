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
support_lang = ['python', 'cpp', 'java', 'javascript', 'php','typescript'
                #  'ruby', 'go',
                #  'html', 'css',
                   ]
not_code = ['tex',  'ini', 'csv' , 'batchfile', 'markdown', 'json', 'prolog' , 'yaml', 'sql', 'powershell', 'dockerfile', 'shell', 'makefile', 'fortran']

def guessLang_classify(block):
    '''
    Classify codeblock language using GuessLang
    '''
    block = str(block)
    block = block.strip()
    if len(block) >0:
        name = guess.language_name(block)
        code = 0
        for  _, value in name[:28]:
            if _.lower() not in support_lang:
                code+=value*1.5
            else:
                code+=value
        notcode = sum(value for _, value in name if _.lower() in not_code)
        # code = sum(value for _, value in name[:28] if _.lower() not in  not_code)

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
    return tp,fp,fn,tn

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
    # print('precision',precision)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # print('recall',recall)
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1




def guessLang_evaluate(block):
    '''
    Classify codeblock language using GuessLang
    '''
    block = str(block)
    block = block.strip()
    if len(block) >0:
        name = guess.language_name(block)
        code = 0
        for  _, value in name[:28]:
            if _.lower() not in support_lang:
                code+=value*1.5
            else:
                code+=value/5
        notcode = sum(value for _, value in name if _.lower() in not_code)
        code_score = code/(code+notcode) 
        not_score = notcode/(code+notcode)
        print(code_score, not_score)
        if code_score > not_score:
    # if len(block) >0:
    #     block = str(block)
    #     block = block.strip()
    #     name = guess.language_name(block)
    #     try:
    #         name = name.lower()
    #         if name in support_lang:
            print(name,':',block)
            return True
        else:
            return False
        # except:
        #     pass
        # code = sum(value for _, value in name[:28] if _.lower() not in  not_code)
        # notcode = sum(value for _, value in name if _.lower() in not_code)
        # code_score = code/(code+notcode) 
        # not_score = notcode/(code+notcode)
        # if code_score > not_score:

        
import re
def preprocss_line(line):
    # Define the pattern to match
    pattern1, pattern2, pattern3,p4 = re.compile(r'!+'), re.compile(r'/\* ///////.*?//////// \*/'), re.compile(r'_+'),re.compile(r'/+')
    output_string = pattern3.sub('', pattern2.sub('', pattern1.sub('', p4.sub('',line))))
    return output_string

def predict_file(file_name):
    print(file_name)
    try:
        with open(file_name, 'r', encoding='utf-8') as source_code:
            code_lines = source_code.readlines()
    except:
        with open(file_name, 'r', encoding='latin-1') as source_code:
            code_lines = source_code.readlines()    
    for line in code_lines:
        if len(line) >0:
            line = str(line)
            line = line.strip()
            line = preprocss_line(line)
            name = guess.language_name(line)
            code = 0
            notcode = sum(value for _, value in name[:20] if _.lower() in not_code)/len(not_code)

            for  _, value in name[:20]:
                if _.lower() in support_lang:
                    code+=value*3.4
                else:
                    code+=value/4
            code= code/20
            # code_score = code/(code+notcode) 
            # not_score = notcode/(code+notcode)
            # print(code, notcode)
            if code > notcode:
            # try:
            #     name = name.lower()
            #     if name in support_lang:
                    print(name,':',line)            
                    print(f'predict: {file_name} contains Source Code')
                    L.configure(text=f'This file contains Source Code', foreground="red")
                    return False # Có chứa Source
            else:
                continue 
                    # return True  # Không chưa Source
            # except:
            #     pass
    L.configure(text=f'# Không chưa Source', foreground="green")

    return True


def evaluate(y_label, y_pred):
    # print('y_label:', y_label)
    # print('predict:', y_pred)
    confusion_matrix(y_label, y_pred)
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    accuracy = (y_label == y_pred).mean()
    print('accuracy', accuracy)
    score = f1_score(y_label,y_pred)
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
    # predict(filename)
import os
from tqdm import tqdm
def openFolder():
    folder_selected = tkinter.filedialog.askdirectory()
    start_time = time.time()
    # print(os.listdir(folder_selected))
    # Using os.walk() 
    ground_truth = []    # 0: Code           1: NoCode
    model_predict = []
    for dirpath, dirs, files in tqdm(os.walk(folder_selected)):  
        for filename in files: 
            fname = dirpath.replace("\\", "/") + '/' + filename
            if predict_file(fname):
                model_predict.append(1.)
                print(1)
            else:
                model_predict.append(0)
                print(0)

            if 'NoCode' in fname: 
                ground_truth.append(1.)
            else:
                ground_truth.append(0)
    evaluate(ground_truth, model_predict)
    print(time.time() - start_time)

    # Change label contents


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
    tk.Button(root, text = "Open Folder", command = openFolder).pack( anchor = tk.W) 
    tk.Label(root,text="Result").pack(anchor=tk.W)
    L = tk.Label(root, text="Hello!", relief="sunken", bg = "light yellow", justify="left")
    L.pack()


    root.mainloop()