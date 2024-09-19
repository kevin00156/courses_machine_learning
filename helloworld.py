import cv2

# 讀取圖像
image = cv2.imread('./image/lenna.jpeg')

# 顯示圖像
cv2.imshow('Image', image)

# 等待任意鍵按下後關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()