import cv2
import numpy as np
import matplotlib.pylab as plt

#get image path
imgPath = "./image/apple_grayscale_and_binarization"
imgName = "original_apple.png"
imgOriginalApple = cv2.imread(f"{imgPath}/{imgName}")

#show original image
cv2.imshow("OriginalApple",imgOriginalApple)
cv2.waitKey(0)
cv2.destroyAllWindows()

#show grayscale image
imgGrayApple = cv2.cvtColor(imgOriginalApple, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_image",imgGrayApple)
cv2.waitKey(0)
cv2.destroyAllWindows()

#transform grayscale image
rtn, imgBinarizationApple = cv2.threshold(imgGrayApple, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binary",imgBinarizationApple)
cv2.waitKey(0)
cv2.destroyAllWindows()

#get a pixel data from gray_image
pixel_data = imgOriginalApple[100,100]
print(pixel_data)

print(imgOriginalApple.shape)
print(imgOriginalApple.size)
print(imgOriginalApple.dtype)


#分割原始圖像的R,G,B亮度，並分別轉換為灰階圖像
R_map = imgOriginalApple[:,:,2]
G_map = imgOriginalApple[:,:,1]
B_map = imgOriginalApple[:,:,0]

plt.figure(figsize=(12,12))

plt.subplot(1,3,1)
plt.axis("off")
plt.imshow(R_map, 'gray')

plt.subplot(1,3,2)
plt.axis("off")
plt.imshow(G_map, 'gray')

plt.subplot(1,3,3)
plt.axis("off")
plt.imshow(B_map, 'gray')

plt.show()

#做各種的thershould練習
ret,thresh1 = cv2.threshold(imgGrayApple,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(imgGrayApple,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(imgGrayApple,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(imgGrayApple,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(imgGrayApple,127,255,cv2.THRESH_TOZERO_INV)

plt.figure(figsize=(12,12))

plt.subplot(2,3,1)
plt.axis("off")
plt.imshow(thresh1, 'gray')

plt.subplot(2,3,2)
plt.axis("off")
plt.imshow(thresh2, 'gray')

plt.subplot(2,3,3)
plt.axis("off")
plt.imshow(thresh3, 'gray')

plt.subplot(2,3,4)
plt.axis("off")
plt.imshow(thresh4, 'gray')

plt.subplot(2,3,5)
plt.axis("off")
plt.imshow(thresh5, 'gray')
plt.show()