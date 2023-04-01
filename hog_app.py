import pickle
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from skimage.feature import hog


def upload_file():
    global photo_image
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes = f_types)
    photo_image = ImageTk.PhotoImage(file = filename)
    b2 = tk.Button(window, image = photo_image) 
    b2.grid(row = 3, column = 1)
    
    image = Image.open(filename)
    feature, hog_img = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    feature = np.array(feature).flatten().reshape(1, -1)

    predicted = classifier.predict(feature)[0]
    predicted = 'Female' if predicted == 0 else 'Male'
    print('predicted', predicted)
    l2.config(text = f'Predicted: {predicted}')
    
    

if __name__ == "__main__":
    
    filename = "/Users//Desktop/project/model/hog_model.pickle"

    # load model
    classifier = pickle.load(open(filename, "rb"))
    
    
    print(dir(classifier))
    
    window = tk.Tk()
    window.geometry("400x300")
    window.title('Hog Gender Classification')

    font = ('times', 18, 'bold')
    l1 = tk.Label(
        window,
        text  = 'Gender Classification',
        width = 30,
        font  = font)  
    l1.grid(row = 1, column = 1)

    b1 = tk.Button(
        window, 
        text    = 'Upload File',
        width   = 20,
        command = lambda:upload_file())
    b1.grid(row = 2, column = 1) 

    l2 = tk.Label(
        window,
        text  = '',
        width = 30,
        font  = font)  
    l2.grid(row = 4, column = 1)
    
    l3 = tk.Label(
        window,
        text  = '',
        width = 30,
        font  = font)  
    l3.grid(row = 5, column = 1)
    
    window.mainloop()
