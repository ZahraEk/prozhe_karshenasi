import cv2
import numpy as np
import matplotlib.pyplot as plt

#khandane aks
img = cv2.imread('images/lena.bmp', 0)

#mask prewitt dar rastaye ofoghi
array_h = np.array([[1,0,-1],
                    [1,0,-1],
                    [1,0,-1]])

#mask prewitt dar rastaye amodi
array_v = np.array([[1,1,1],
                    [0,0,0],
                    [-1,-1,-1]])

#convolve kardan maskha dar tasvir
horizental = cv2.filter2D(src=img, ddepth=-1, kernel=array_h)
vertical = cv2.filter2D(src=img, ddepth=-1, kernel=array_v)

#be dast avardn filter prewitt da rastaye ofoghi va amodi
prewitt_h = np.asarray(np.clip(horizental, 0, 255), dtype="uint8")
prewitt_v = np.asarray(np.clip(vertical, 0, 255), dtype="uint8")
prewitts = prewitt_h + prewitt_v

#save kardan khoroji
prewitt_res = cv2.imwrite('images/prewitt_result.jpg', prewitts)

#namayeshe tasavir
f, subplt = plt.subplots(1, 4, figsize=(16, 4))
subplt[0].imshow(img, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(prewitt_h, cmap='gray')
subplt[1].set_title("Horizental Prewitt")
subplt[2].imshow(prewitt_v, cmap='gray')
subplt[2].set_title("Vertical Prewitt")
subplt[3].imshow(prewitts, cmap='gray')
subplt[3].set_title("Prewitts")
plt.show()