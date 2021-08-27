import cv2
import numpy as np
import matplotlib.pyplot as plt

#khandane aks
img = cv2.imread("images/lena.bmp",0)

#mask sobel dar rastaye ofoghi
array_h = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
#mask sobel dar rastaye amodi
array_v = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])

#convolve kardan maskha dar tasvir
horizental = cv2.filter2D(src=img, ddepth=-1, kernel=array_h)
vertical = cv2.filter2D(src=img, ddepth=-1, kernel=array_v)

#be dast avardn filter sobel da rastaye ofoghi va amodi
sobel_h = np.asarray(np.clip(horizental, 0, 255), dtype="uint8")
sobel_v = np.asarray(np.clip(vertical, 0, 255), dtype="uint8")
sobels = sobel_h + sobel_v

#save kardan khoroji
sobel_res = cv2.imwrite('images/sobel_result.jpg', sobels)

#namayeshe tasavir
f, subplt = plt.subplots(1, 4, figsize=(16, 4))
subplt[0].imshow(img, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(sobel_h, cmap='gray')
subplt[1].set_title("Horizental Sobel")
subplt[2].imshow(sobel_v, cmap='gray')
subplt[2].set_title("Vertical Sobel")
subplt[3].imshow(sobels, cmap='gray')
subplt[3].set_title("Sobels")
plt.show()