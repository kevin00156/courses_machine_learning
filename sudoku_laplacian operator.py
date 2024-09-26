import numpy as np
import cv2
from matplotlib import pyplot as plt

#讀取圖像：sudoku-original.png
imgPath = "image/"
imgName = "sudoku-original.png"
imgSudoku = cv2.imread(f"{imgPath}/{imgName}")

#把圖像拿去做sobel運算
x = cv2.Sobel(imgSudoku, cv2.CV_16S, 1, 0)
y = cv2.Sobel(imgSudoku, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
mag = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(imgSudoku,cmap='gray')
plt.show()

plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(imgSudoku,cmap='gray')
plt.show()

#執行拉普拉斯影像轉換
gray_lap = cv2.Laplacian(imgSudoku, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(gray_lap)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(imgSudoku,cmap='gray')
plt.show()

plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(dst,cmap='cividis')
plt.show()