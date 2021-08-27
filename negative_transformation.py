import cv2
import matplotlib.pyplot as plt

#khandane aks
org_img = cv2.imread("images/baboon.png", 0)
img = cv2.cvtColor(org_img , cv2.COLOR_BGR2RGB)

#negative kardan tasvir
img_not = cv2.bitwise_not(img)

#save kardan khoroji
mask = cv2.imwrite('images/negative_img.jpg', img_not)

#namayeshe tasavir va histogram an ha
f , subplt = plt.subplots(2,2,figsize=(12,12))
subplt[0,0].imshow(img)
subplt[0,0].set_title("Original_img")
subplt[0,1].imshow(img_not)
subplt[0,1].set_title("Negative_img")

subplt[1,0].hist(img.ravel(), 256)
subplt[1,0].set_title("Histogram of Original_img")
subplt[1,1].hist(img_not.ravel(), 256)
subplt[1,1].set_title("Histogram of Negative_img")
plt.show()

