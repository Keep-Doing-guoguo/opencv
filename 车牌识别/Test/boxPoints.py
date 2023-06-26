import cv2
import numpy as np
# 创建一个旋转矩形
center = (100, 100)
size = (200, 100)
angle = 30
rect = (center, size, angle)

# 计算旋转矩形的四个角点坐标
points = cv2.boxPoints(rect)
print("Points:", points)

# 将浮点型坐标点转换为整数型
points = np.int0(points)

# 绘制旋转矩形
image = np.zeros((200, 200), dtype=np.uint8)
cv2.drawContours(image, [points], 0, 255, 2)

# 显示图像
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
