import cv2
import numpy as np
import matplotlib.pyplot as plt

#khandane aks
org_img = cv2.imread("images/sample_2.jpg")
img= cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

#tabe eslahe gamma
#gamma = 1 => tabdile identity. taghiri dar tasvir ijad nemishavad
#gamma < 1 => tabdile logaritmic. baese roshan tar shodan tasvir mishavad
#gamma > 1 => tabdile nammayi. baese tarik tar shodn tasvir mishavad
def gammaCorrection(src, gamma):

    table = [((i / 255) ** gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

#emale eslahe gamma bar roye tasvir
correction_img = gammaCorrection(img, 2)

#save kardan khoroji
mask = cv2.imwrite('images/gamma_result.jpg', correction_img)

#namayeshe tasavir va histogram an ha
f, subplt = plt.subplots(2,2, figsize=(12,12))
subplt[0,0].imshow(img)
subplt[0,0].set_title("Original_img")
subplt[0,1].hist(img.ravel(), 256)
subplt[0,1].set_title("Histogram of Original_img")

subplt[1,0].imshow(correction_img)
subplt[1,0].set_title("Corrected_img")
subplt[1,1].hist(correction_img.ravel(), 256)
subplt[1,1].set_title("Histogram of corrected_img")
plt.show()







