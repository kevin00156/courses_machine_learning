import cv2
import numpy as np

# 讀取圖像
image = cv2.imread('./image/affine_pratice_image.png')

# 顯示圖像
cv2.imshow('Image_origin', image)

# 等待任意鍵按下後關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()


#指定轉換座標
pts1 = np.float32([[23,206], [172,139], [269,355]])#原始座標
pts2 = np.float32([[0,0], [200,0], [200,200]])#新的scale座標
h, w, c = image.shape#取得原始圖片資訊

#指定轉換方法
M = cv2.getAffineTransform(pts1,pts2)
new_image = cv2.warpAffine(image, M, (w, h))
# 顯示圖像
cv2.imshow('Image_new', new_image)

# 等待任意鍵按下後關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
