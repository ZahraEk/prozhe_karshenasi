import cv2 as cv
import cv2
from matplotlib import pyplot as plt

#khandane aks
img = cv2.imread("images/Statue.png",0)

#hamsan sazi histogram
equ = cv.equalizeHist(img)

#sakhtane CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(img)

#namayeshe tasavir va histogram an ha
f, subplt = plt.subplots(2,3 , figsize=(20,10))
subplt[0,0].imshow(equ, cmap="gray")
subplt[0,0].set_title("Equalized_img")
subplt[0,1].imshow(img, cmap="gray")
subplt[0,1].set_title("Orginal_img")
subplt[0,2].imshow(cl, cmap="gray")
subplt[0,2].set_title("Clahe_img")

subplt[1,0].hist(equ.flatten(),256,[0,256])
subplt[1,0].set_title("Histogram of Equalized_img")
subplt[1,1].hist(img.flatten(),256,[0,256])
subplt[1,1].set_title("Histogram of Orginal_img")
subplt[1,2].hist(cl.flatten(),256,[0,256])
subplt[1,2].set_title("Histogram of Clahe_img")
plt.show()


