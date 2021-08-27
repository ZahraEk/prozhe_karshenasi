import cv2
from matplotlib import pyplot as plt

#khandane aks
img = cv2.imread("images/IMG_0001_resize.jpg", 0)

#sakhtane CLAHE
clahe = cv2.createCLAHE(clipLimit=70.0, tileGridSize=(2,2))
cl = clahe.apply(img)

#save kardan tasvir hamsansazi shode
clahe_img = cv2.imwrite('images/SilandrCl0001.jpg', cl)

#namayeshe tasavir va histogram an ha
f, subplt = plt.subplots(2,2 , figsize=(16,10))
subplt[0,0].imshow(img, cmap="gray")
subplt[0,0].set_title("Orginal_img")
subplt[0,1].imshow(cl, cmap="gray")
subplt[0,1].set_title("Clahe_img")
subplt[1,0].hist(img.flatten(),256,[0,256])
subplt[1,0].set_title("Histogram of Orginal_img")
subplt[1,1].hist(cl.flatten(),256,[0,256])
subplt[1,1].set_title("Histogram of Clahe_img")
plt.show()







