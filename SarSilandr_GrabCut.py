import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#khandane aks
img = cv.imread("images/IMG_2933_resize.jpg")

#ijad mask
mask = np.zeros(img.shape[:2],np.uint8)

#ijad asaraye bgModel va fgModel
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#ijad mostatil
#silandr 0010 : rect = (10,24,494,260)
#silandr 0492 : rect = (3,20,545,300)
#silandr 0082 : rect = (3,20,493,255)
#silandr 0001 : rect = (0,50,500,255)
#silandr 2933 :
rect = (3,3,505,450)

#emale grabcut
cv.grabCut(img,mask,rect,bgdModel,fgdModel,2,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

#namayeshe aks
img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show()

#save kardan tasvir jodasazi shode(marhaleye aval)
Siland_GrabCut1 = cv.imwrite('images/SilandrCut2933.jpg', img)

##khandane maski ke tavasato khodeman ijad shode
#newmask = cv.imread('images/mask.jpg',0)

##har ja k ba sefid moshakhas shode hatman pishzamine ast, taghir mask=1
##har ja k ba meshki moshakhas shode hatmn paszamine asr, taghir mask=0
#mask[newmask == 0] = 0
#mask[newmask == 255] = 1

##emale grabcut
#mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
#mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#img = img*mask[:,:,np.newaxis]

##namayeshe tasvir
#img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#plt.imshow(img1)
#plt.show()

#save kardan tasvir jodasazi shode(marhaleye dovom)
##Siland_GrabCut = cv.imwrite('images/Siland_GrabCut.jpg', img)



