import cv2 as cv
from matplotlib import pyplot as plt

#khandane aksha
img_cut = cv.imread("images/SilandrCut0492.jpg")
img_cut1 = cv.cvtColor(img_cut, cv.COLOR_BGR2RGB)

img_gray = cv.imread('images/IMG_0492_resize.jpg',0)

org_img = cv.imread('images/IMG_0492.JPG')
org_img1 = cv.cvtColor(org_img, cv.COLOR_BGR2RGB)

#emale threshold bar royr aksha
#silandr 0010 : ret,trs = cv.threshold(img_gray,140,255,cv.THRESH_BINARY)
#silandr 0492 :
ret,trs = cv.threshold(img_gray,165,255,cv.THRESH_BINARY_INV)
#silandr 0082 : ret,trs = cv.threshold(img_gray,155,250,cv.THRESH_BINARY)
#silandr 0001 : ret,trs = cv.threshold(img_gray,120,250,cv.THRESH_BINARY_INV)
#silandr 2933 : ret,trs = cv.threshold(img_gray,163,250,cv.THRESH_BINARY)

#emale mask threshold bar roye aks cut shode
res = cv.bitwise_and(img_cut1, img_cut1, mask=trs)
res2 = cv.cvtColor(res, cv.COLOR_BGR2RGB)

#save kardane aksha
mask = cv.imwrite('images/SilandrMask0492.jpg', trs)
result = cv.imwrite('images/SilandrRes0492.jpg', res2)

#namayeshe aksha
f, subplt = plt.subplots(2,2,figsize=(30,18))
subplt[0,0].imshow(org_img1)
subplt[0,0].set_title("Orginal Img")
subplt[0,1].imshow(img_cut1,cmap="gray")
subplt[0,1].set_title("Cut Img")
subplt[1,0].imshow(trs, cmap="gray")
subplt[1,0].set_title("Threshold Img")
subplt[1,1].imshow(res)
subplt[1,1].set_title("Result Img")
plt.show()