import cv2
import numpy as np


img = cv2.imread("TP.jpg",cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erreur chargement de l'image")
    exit(0)


img[:] = img[:] / 3
w,h = img.shape
print(img.shape)
min,max = 255,0




for x in range(w):
    for y in range(h):
        if img[x,y] > max:
            max = img[x,y]
        if min > img[x,y]:
            min = img[x,y]


print("min ",min,"; max : ",max)

new_img = np.zeros(img.shape,dtype=img.dtype)

for x in range(w):
    for y in range(h):
        new_img[x,y] = (img[x,y] - min)*255 / (max - min)



def makeHist(img):
    to_hist = img
    hist = np.zeros((256,1),dtype=np.int32)
    for x in range(w):
        for y in range(h):
            hist[to_hist[x,y]] +=1
    return hist



def makeplot(array,width):
    plot = np.zeros((500,width))
    plot[:,:] = 255
    for x in range(width):
        for y in range(500):
            if( array[x] > y):
                plot[499 - y,x] = 0
    return plot



hist = makeHist(img)
hist2 = cv2.calcHist([new_img],[0],None,[255],[0,255])
cv2.imshow("plot",makeplot(hist,hist.shape[0]))
cv2.imshow("plot2",makeplot(hist2,hist2.shape[0]))
cv2.imshow("img1",img)
cv2.imshow("img2",new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

