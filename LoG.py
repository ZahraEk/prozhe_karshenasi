import cv2
import matplotlib.pyplot as plt

#khandane aks
img = cv2.imread("images/lena.bmp",0)

#emale filtere gaussian baraye kaheshe noise
gaussian = cv2.GaussianBlur(img, (7,7), 0)

#emale filtere laplacian bar rooye tasvir hamvar sazi shode
LoG = cv2.Laplacian(gaussian, cv2.CV_8U, ksize=5)

#save kardan khoroji
LoG_res = cv2.imwrite('images/LoG_result.jpg', LoG)

#namayeshe tasavir
f, subplt = plt.subplots(1,2,figsize=(16,8))
subplt[0].imshow(img, cmap='gray')
subplt[0].set_title("Orginal Img")
subplt[1].imshow(LoG, cmap='gray')
subplt[1].set_title("LoG Filter")

plt.show()
