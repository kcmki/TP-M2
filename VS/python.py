import cv2
import numpy as np

img = cv2.imread("TP.jpg",cv2.IMREAD_COLOR)
if img is None:
    print("Erreur")
    exit(0)



result = np.array(255 - img)

fade = np.zeros((500,500),np.uint8)

for x in range(500):
    for y in range(500):
        fade[x,y] = (x+y) * (255) / (fade.shape[0]+fade.shape[1])


print(fade)

cv2.imshow("fade",fade)


cv2.waitKey(0)
