import cv2
import matplotlib.pyplot as plt

#khandane aks
org_img = cv2.imread('images/cameraman.png',0)
img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

#emale threshold
ret,trs = cv2.threshold(org_img,100,255,cv2.THRESH_BINARY)

#namayeshe tasvir va histogram an
f, subplt = plt.subplots(2,2)

subplt[0,0].imshow(img, cmap="gray")
subplt[0,0].set_title("Original")
subplt[0,1].imshow(trs, cmap="gray")
subplt[0,1].set_title("Threshold")

subplt[1,0].hist(img.ravel(), 256)
subplt[1,0].set_title("Histogram of Original_img")
subplt[1,1].hist(trs.ravel(), 256)
subplt[1,1].set_title("Histogram of Negative_img")

plt.show()