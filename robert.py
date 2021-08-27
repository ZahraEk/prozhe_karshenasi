import cv2
import numpy as np
import matplotlib.pyplot as plt

#khandane aks
img = cv2.imread('images/lena.bmp', 0)

#mask robert dar rastaye ofoghi
array_h = np.array([[0,0,0],
                    [0,0,1],
                    [0,-1,0]])

#mask robert dar rastaye amodi
array_v = np.array([[0,0,0],
                    [0,1,0],
                    [0,0,-1]])

#convolve kardan maskha ba tasvir
horizental = cv2.filter2D(src=img, ddepth=-1, kernel=array_h)
vertical = cv2.filter2D(src=img, ddepth=-1, kernel=array_v)

#be dast avardn filter robert da rastaye ofoghi va amodi
robert_h = np.asarray(np.clip(horizental, 0, 255), dtype="uint8")
robert_v = np.asarray(np.clip(vertical, 0, 255), dtype="uint8")
roberts = robert_h + robert_v

#save kardan khoroji
robert_res = cv2.imwrite('images/robert_result.jpg', roberts)

#namayeshe tasavir
f, subplt = plt.subplots(1, 4, figsize=(16, 4))
subplt[0].imshow(img, cmap='gray')
subplt[0].set_title("Original Image")
subplt[1].imshow(robert_h, cmap='gray')
subplt[1].set_title("Horizental Robert")
subplt[2].imshow(robert_v, cmap='gray')
subplt[2].set_title("Vertical Robert")
subplt[3].imshow(roberts, cmap='gray')
subplt[3].set_title("Roberts")
plt.show()