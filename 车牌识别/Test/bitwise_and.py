import cv2
def image_and(image,mask):#输入图像和掩膜
	area = cv2.bitwise_and(image,image,mask=mask)  #mask=mask表示要提取的区域
	cv2.imshow("area",area)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return area
