import cv2
import numpy as np 
import sqlite3
import os
from PIL import ImageTk
import sklearn as ss
print(ss.__version__)
from tkinter import *
from matplotlib import pyplot as plt
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
f1 = 'GradientBoostingClassifier.sav'  #0.8470370370370371
f2 = 'MultinomialNB.sav'   #0.8723280423280423
f3 = 'LogisticRegression.sav'  #0.6770370370370371
f4 = 'SVC.sav'  #0.7502645502645502
import tkinter.messagebox
#global var
from PIL import ImageTk,Image
global screen
import pickle
loaded_model = pickle.load(open(f4, 'rb'))
tf = pickle.load(open("feature.pkl", 'rb'))

def main_screen():
    global screen3
    screen3=Tk()
    global inputsong
    scroll = Scrollbar(screen3)
    scroll.pack(side=RIGHT, fill=Y)
   
    inputsong = Text(screen3, height = 23, width = 75,bg = "light yellow",wrap=NONE, yscrollcommand=scroll.set)
    scroll.config(command=inputsong.yview)
    inputsong.pack()
    Label(screen3, text="", bg="blue").pack()
    b1 =  Button(screen3,text="Sentiment Prediction",height=2,width=50,bg="Black",fg="blue",font=("Arial Bold", 15), command = lambda:Take_input())
    b1.pack()

    Label(screen3, text="", bg="red").pack()
    b2 =  Button(screen3,text="Graph Analysis",height=2,width=50,bg="Black",fg="red",font=("Arial Bold", 15), command=graph)
    b2.pack()

   
 
    Label(screen3, text="", bg="green").pack()
    b5 =  Button(screen3,text="Exit",height=2,width=50,bg="Black",fg="green",font=("Arial Bold", 15),command=screen3.destroy)
    b5.pack()
def Take_input():
    INPUT = inputsong.get("1.0", "end-1c")
    ResultAnswer = loaded_model.predict(tf.transform([INPUT]).toarray())
    print(ResultAnswer) #[[0]],    [[1]], [[2]], [[3]]
    # 'angry': 0 ,'sad':1, 'relaxed':2 , 'happy': 3

    if ResultAnswer[0] == 0:
        print("Angry Song")
        tkinter.messagebox.showinfo('Sentiment Analysis', 'Uploaded lyrics have ANGRY Sentiment.')
    
    elif ResultAnswer[0] == 1:
        print("Sad Song")
        tkinter.messagebox.showinfo('Sentiment Analysis', 'Uploaded lyrics have SAD Sentiment.')
    
    elif ResultAnswer[0] == 2:
        print("Relaxed Song")
        tkinter.messagebox.showinfo('Sentiment Analysis', 'Uploaded lyrics have RELAXED Sentiment.')
    
    else:
        print("Happy Song")
        tkinter.messagebox.showinfo('Sentiment Analysis', 'Uploaded lyrics have HAPPY Sentiment.')
import cv2
def graph():
    image = cv2.imread("h.png")
    cv2.imshow("Accuraccy Analysis",image)
main_screen()

