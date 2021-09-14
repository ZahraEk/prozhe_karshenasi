import cv2
import matplotlib.pyplot as plt

#khandane aks
org_img = cv2.imread('images/cameraman.png',0)
img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

#emale threshold
ret,trs = cv2.threshold(org_img,100,255,cv2.THRESH_BINARY)

#namayeshe tasvir va histogram an
f, subplt = plt.subplots(1,2)

subplt[0].imshow(img, cmap="gray")
subplt[0].set_title("Original")
subplt[1].imshow(trs, cmap="gray")
subplt[1].set_title("Threshold")

plt.show()