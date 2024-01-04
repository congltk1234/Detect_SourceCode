import tkinter as tk

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
    print(f"Guesslang: {str(guesslang_check.get())},\nCodeBERT: {str(codebert_check.get())},\nMode: {var.get()}")  
    print(f'Input text: {scroll.get("1.0","end-1c")}')
    L.configure(text=f'{scroll.get("1.0","end-1c")[:100]}')

if __name__ == '__main__':
    root = tk.Tk()
    scroll = ScrollText(root)
    scroll.insert(tk.END, "HEY" + 20*'\n')
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


    model_select = tk.Label(root, text="Model selection:").pack( anchor = tk.W)
    guesslang_check = tk.IntVar()
    codebert_check = tk.IntVar()

    tk.Checkbutton(root, text="Guesslang", variable=guesslang_check).pack( anchor = tk.W)
    tk.Checkbutton(root, text="CodeBERT", variable=codebert_check).pack( anchor = tk.W)
    tk.Button(root, text='Predict', command=var_states).pack( anchor = tk.W)

    # •Label():
    tk.Label(root,text="Result").pack(anchor=tk.W)
    L = tk.Label(root, text="Hello!", relief="sunken", bg = "light yellow")
    L.pack()


    root.mainloop()