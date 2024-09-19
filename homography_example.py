import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('./image/homography_pratice_image.png')

# 顯示圖像
cv2.imshow('Image_origin', image)

# 等待任意鍵按下後關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()


#指定轉換座標
pts1 = np.float32([[283,116], [404,168], [141,313], [24,209]])
pts2 = np.float32([[100,100], [300,100], [300,400], [100,400]])


#指定轉換方法
M = cv2.getPerspectiveTransform(pts1, pts2)
new_image = cv2.warpPerspective(image, M, (500, 500))
# 顯示圖像
cv2.imshow('Image_new', new_image)

# 等待任意鍵按下後關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
