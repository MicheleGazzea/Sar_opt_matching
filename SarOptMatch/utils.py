import time
import tkinter as tk
from tkinter import filedialog



def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        print("Time for " + str(func.__name__) + " is "+ str(time.time() - start))
        return output
    return wrapper



def confirmCommand(text):
    text = text + " (y/n)"
    check = input(text)
    if check.lower() in ["y", "yes"]:
        return True
    else:
        return False
    

    
def open_file(title = ""):
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()   # this supress the tk window
    
    window.update()
    filename = filedialog.askopenfilename(parent=window,
                                      initialdir="",
                                      title=title,
                                      filetypes = (("model weights", ".h5"),
                                                   ("All files", "*")))
    window.destroy()
    # Here, window.wm_attributes('-topmost', 1) and "parent=window" argument 
    # help open the dialog box on top of other windows
    return filename