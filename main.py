import tkinter as tk
import function
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import math
import time


# 打开图片
def openpicture():
    global picture1, filename  # 需声明为全局变量
    filename = filedialog.askopenfilename()
    image = Image.open(filename)
    picture1 = ImageTk.PhotoImage(image)
    canvas1 = tk.Canvas(upfm1, height=450, width=600)
    canvas1.grid(row=0, column=0)
    canvas1.create_image(300, 225, image=picture1)


# 图片去雾
def dehaze_func():
    global picture2
    img = cv.imread(filename)
    img_arr = np.array(img / 255.0)  # 归一化
    img_min = function.darkchannel(img_arr)  # 计算每个通道的最小值
    img_guided = function.guided_filter(img_arr, img_min, r=75, eps=0.001)
    t, A = function.select_bright(img_min, img, w=0.95, t0=0.1, V=img_guided)
    dehaze = function.repair(img_arr, t, A)
    dehaze = dehaze * 255
    cv.imwrite('dehaze.jpg', dehaze)
    image = Image.open('dehaze.jpg')
    picture2 = ImageTk.PhotoImage(image)
    canvas2 = tk.Canvas(upfm2, height=450, width=600)
    canvas2.grid(row=0, column=1)
    canvas2.create_image(300, 225, image=picture2)  # 宽 * 高


# 图片增强
def picexhance():
    global picture3
    img = cv.imread('dehaze.jpg')
    t1 = time.time()
    img = img / 255
    Imax = np.max(img)
    I1 = Imax / np.log(Imax + 1) * np.log(img + 1)
    I2 = 1 - np.exp(-img)
    I3 = (I1 + I2) / (2 + I1 * I2)
    h, w, c = I3.shape
    I4 = np.zeros_like(I3)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                I4[i, j, k] = math.erf(2 * math.atan(math.exp(I3[i, j, k])) - 0.5 * I3[i, j, k])
    I5 = (I4 - np.min(I4)) / (np.max(I4) - np.min(I4))
    t2 = time.time()
    var.set('time : '+str(t2-t1)+'s')
    I5 = I5 * 255
    cv.imwrite('picexhance.jpg', I5)
    image = Image.open('picexhance.jpg')
    picture3 = ImageTk.PhotoImage(image)
    canvas3 = tk.Canvas(upfm3, height=450, width=600)
    canvas3.grid(row=0, column=2)
    canvas3.create_image(300, 225, image=picture3)


# 布局
root = tk.Tk()
root.title("程序")
var = tk.StringVar()
upfm1 = tk.Frame(root, height=450, width=600).grid(row=0, column=0)
upfm2 = tk.Frame(root, height=450, width=600).grid(row=0, column=1)
upfm3 = tk.Frame(root, height=450, width=600).grid(row=0, column=2)
downfm = tk.Frame(root, height=100, width=1800).grid(row=1, column=0, rowspan=2, columnspan=3)
imgBtn1 = tk.PhotoImage(file=r'button/openpic.png')
button1 = tk.Button(downfm, image=imgBtn1, command=openpicture, borderwidth=0)
button1.grid(row=1, column=0)
imgBtn2 = tk.PhotoImage(file=r'button/dehaze.png')
button2 = tk.Button(downfm, image=imgBtn2, command=dehaze_func, borderwidth=0)
button2.grid(row=1, column=1)
imgBtn3 = tk.PhotoImage(file=r'button/exhance.png')
button3 = tk.Button(downfm, image=imgBtn3, command=picexhance, borderwidth=0)
button3.grid(row=1, column=2)
time_lable = tk.Label(downfm, textvariable=var)
time_lable.grid(row=2, column=2, sticky=tk.S+tk.E)
tk.mainloop()