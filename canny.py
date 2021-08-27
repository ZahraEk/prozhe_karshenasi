import cv2
import matplotlib.pyplot as plt

#khandane aks
img = cv2.imread("images/lena.bmp",0)

#emale filtere canny
canny_img = cv2.Canny(img,50,120)

#save kardan khoroji
canny_res = cv2.imwrite('images/canny_result.jpg', canny_img)

#namayeshe tasavir
f, subplt = plt.subplots(1,2,figsize=(16,8))
subplt[0].imshow(img, cmap='gray')
subplt[0].set_title("Orginal Img")
subplt[1].imshow(canny_img, cmap='gray')
subplt[1].set_title("Canny Img")
plt.show()
