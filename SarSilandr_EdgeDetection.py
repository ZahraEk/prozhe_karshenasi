import cv2
from matplotlib import pyplot as plt

#khandane aks
img = cv2.imread("images/IMG_0001_resize.jpg",0)

#emale filtere gaussian baraye kaheshe noise
gaussian = cv2.GaussianBlur(img, (3,3), 0)

#emale filtere laplacian bar rooye tasvir hamvar sazi shode
LoG = cv2.Laplacian(gaussian, cv2.CV_8UC1, ksize=5)

#namayeshe tasavir
f, subplt = plt.subplots(1,2,figsize=(16,5))
subplt[0].imshow(img, cmap='gray')
subplt[0].set_title("Orginal Img")
subplt[1].imshow(LoG, cmap='gray')
subplt[1].set_title("Edged Img")
plt.show()

#save kardan tasvir labe yabi sjode
Edged_img = cv2.imwrite('images/Edged_img2.jpg', LoG)




