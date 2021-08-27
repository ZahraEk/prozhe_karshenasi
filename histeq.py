import cv2 as cv
from matplotlib import pyplot as plt

#khandane aks
img = cv.imread("images/sample_1.jpg",0)

#hamsan sazi histogram
equ = cv.equalizeHist(img)

equ_res = cv.imwrite('images/Equalized_img.jpg', equ)

#namayeshe tasavir va histogram an ha
f, subplt = plt.subplots(2,2 , figsize=(12,14))
subplt[0,0].imshow(img, cmap="gray")
subplt[0,0].set_title("Original_img")
subplt[0,1].imshow(equ, cmap="gray")
subplt[0,1].set_title("Equalized_img")

subplt[1,0].hist(img.flatten(),256,[0,256])
subplt[1,0].set_title("Histogram of Original_img")
subplt[1,1].hist(equ.flatten(),256,[0,256])
subplt[1,1].set_title("Histogram of Equalized_img")
plt.show()

