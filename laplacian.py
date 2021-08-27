import cv2
import matplotlib.pyplot as plt

#khandane aks
img = cv2.imread("images/lena.bmp",0)

#emale filtere laplacian
laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=5)

#save kardan khoroji
laplacian_res = cv2.imwrite('images/laplacian_result.jpg', laplacian)

#namayeshe tasavir
f, subplt = plt.subplots(1,2,figsize=(16,8))
subplt[0].imshow(img, cmap='gray')
subplt[0].set_title("Orginal Img")
subplt[1].imshow(laplacian, cmap='gray')
subplt[1].set_title("Laplacian Filter")
plt.show()











