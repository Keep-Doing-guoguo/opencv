# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
#默认的时候是1代表彩色图，为0的时候代表灰色图。
img = cv2.imread('/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject2/Opencv_Learning/车牌识别/Test/20130622193552343.jpg', 0)

# OpenCV定义的结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 腐蚀图像
eroded = cv2.erode(img, kernel)
# 显示腐蚀后的图像
cv2.imshow("Eroded Image", eroded)
#图像进行先腐蚀后膨胀，ds，代表的是开运算，去除黑色中的白色点
# 膨胀图像
dilated = cv2.dilate(img, kernel)
# 显示膨胀后的图像
cv2.imshow("Dilated Image", dilated)
# 原图像
cv2.imshow("Origin", img)

#一个开运算和腐蚀-膨胀的效果是相同的在这里进行对比
eroded = cv2.erode(img,kernel)
dilated = cv2.dilate(eroded,kernel)
cv2.imshow('Eroded-Swell Image',dilated)

open = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('Open Image',open)


# NumPy定义的结构元素
NpKernel = np.uint8(np.ones((3, 3)))
Nperoded = cv2.erode(img, NpKernel)
# 显示腐蚀后的图像
cv2.imshow("Eroded by NumPy kernel", Nperoded)
cv2.waitKey(0)
cv2.destroyAllWindows()