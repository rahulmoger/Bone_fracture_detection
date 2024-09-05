import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2

import numpy as np
# load the trained model to classify sign
from keras.models import load_model

model = load_model('my_model.h5')

# dictionary to label all traffic signs class.
classes = {
           0: 'Normal',
           1: 'Abnormal'
           }

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Bone Fracture Detection')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path).convert('RGB')
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    print('pred',pred)
    sign = classes[pred + 1]
    #print(sign)

    image = cv2.imread(file_path)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, img1) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    n_white_pix = np.sum(img1 == 255)

    if n_white_pix < 2000:
        sign = 'Normal'
    else:
        sign = 'Abnormal'

    label.configure(foreground='#011638', text=sign)
    if sign=='Abnormal':
        img = cv2.imread(file_path)

        # Find the contours
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 60, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

        # For each contour, find the bounding rectangle and draw it
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        # Finally show the image
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Bone Fracture Detection", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
