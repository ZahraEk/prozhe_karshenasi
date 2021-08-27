import cv2 as  cv
import numpy as np
from matplotlib import pyplot as plt

#khandane aks
org_img = cv.imread("images/Siland_GrabCut1.jpg", 0)
img = cv.cvtColor(org_img, cv.COLOR_BGR2RGB)

#emale filtere median
gray = cv.medianBlur(org_img, 3)

#shensayi dayere ha
rows = gray.shape[0]
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                          param1=50, param2=50,
                          minRadius=50, maxRadius=90)

#keshidane dayere ha
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        #markaze dayere
        cv.circle(img, center, 1, (0, 0, 0), 85)
        #marz dayere
        radius = i[2]

#save kardan khoroji
mask = cv.imwrite('images/HC.jpg', img)

#namayeshe natije
plt.imshow(img)
plt.show()
