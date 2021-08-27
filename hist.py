import cv2
import matplotlib.pyplot as plt

#khandane aks
org_img = cv2.imread('images/baboon.png')
img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

#namayeshe tasvir va histogram an
f, subplt = plt.subplots(1,2)

subplt[0].imshow(img)
subplt[0].set_title("Org Img")
subplt[1].hist(img.flatten(),256,[0,256])
subplt[1].set_title("Histogram of Org Img")
plt.show()