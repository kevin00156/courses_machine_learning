import cv2
import numpy as np
import matplotlib.pylab as plt

#讀取圖像：lenna.jpeg
imgPath = "image/"
imgName = "lenna.jpeg"
imgLenna = cv2.imread(f"{imgPath}/{imgName}")

imgLenna = cv2.cvtColor(imgLenna, cv2.COLOR_BGR2RGB)
imgLennaGray = cv2.cvtColor(imgLenna, cv2.COLOR_RGB2GRAY)

#設定plt的輸出大小
plt.figure(figsize=(8,8))

#設定子plot 要show原始圖像
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(imgLenna)

#設定子plot 要show出灰階圖像
plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(imgLennaGray,cmap='gray')
plt.show()

#執行正常化
kernel = np.ones((11,11),np.float32)/121
blurred_img = cv2.filter2D(imgLennaGray,-1,kernel)

plt.axis("off")
plt.imshow(blurred_img,cmap='gray')
plt.show()

#執行高斯模糊化
gaussblurred_img = cv2.GaussianBlur(imgLennaGray,(11,11),0)

plt.axis("off")
plt.imshow(gaussblurred_img,cmap='gray')
plt.show()

#執行中值濾波器
median = cv2.medianBlur(imgLennaGray,5)
plt.axis("off")
plt.imshow(median,cmap='gray')
plt.show()

#執行索伯Sobel運算子(是拿來做邊緣檢測的前處理之一)
gx = cv2.Sobel(imgLennaGray, cv2.CV_16S, 1, 0)
gy = cv2.Sobel(imgLennaGray, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(gx)
plt.axis("off")
plt.imshow(absX,cmap='gray')
plt.show()

absY = cv2.convertScaleAbs(gy)
plt.axis("off")
plt.imshow(absY,cmap='gray')
plt.show()

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
plt.axis("off")
plt.imshow(dst,cmap='gray')
plt.show()


