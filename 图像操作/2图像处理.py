import cv2#opencv读取图片的格式是BGR
import numpy as np
import matplotlib.pyplot as plt#Matplotlib是RGB

#============================灰度图============================
img = cv2.imread('./cat.jpg')
#色彩空间图的转换
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(img_gray.shape)
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#============================HSV============================
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# cv_show('hsv',hsv)
# H - 色调（主波长）。
# S - 饱和度（纯度/颜色的阴影）。
# V值（强度）

#============================图像阈值============================
ret,thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

#============================图像平滑============================
#https://blog.csdn.net/ShaoDu/article/details/96429733?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-96429733-blog-121455266.pc_relevant_multi_platform_whitelistv4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-96429733-blog-121455266.pc_relevant_multi_platform_whitelistv4&utm_relevant_index=2

img = cv2.imread('lenaNoise.png')#这个是一个有噪声的图片
# cv_show('img',img)
#均值滤波
#简单的平均卷积操作
blur = cv2.blur(img,(3,3))
# 方框滤波
# 基本和均值一样，可以选择归一化
box = cv2.boxFilter(img,-1,(3,3), normalize=True)
# 方框滤波
# 基本和均值一样，可以选择归一化,容易越界
box = cv2.boxFilter(img,-1,(3,3), normalize=False)
# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
aussian = cv2.GaussianBlur(img, (5, 5), 1)#X方向方差主要控制权重。
# 中值滤波
# 相当于用中值代替
median = cv2.medianBlur(img, 5)  # 中值滤波
# 展示所有的
res = np.hstack((img,blur,aussian,median))
# cv_show('res',res)

#============================形态学-腐蚀操作============================
img = cv2.imread('./dige.png')
# cv_show('img',img)
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations=1)
res = np.hstack((img,erosion))
#cv_show('res',res)
pie= cv2.imread('./pie.png')
# cv_show('pie',pie)
kernel = np.ones((30,30),np.uint8)
erosion1 = cv2.erode(img,kernel,iterations=1)
erosion2 = cv2.erode(img,kernel,iterations=2)
erosion3 = cv2.erode(img,kernel,iterations=3)
res = np.hstack((erosion1,erosion2,erosion3))
# cv_show('res',res)
#============================形态学-膨胀操作============================
img = cv2.imread('./dige.png')
# cv_show('img',img)
kernel = np.ones((3,3),np.uint8)
dige_erosion = cv2.erode(img,kernel,iterations = 1)
res = np.hstack((img,dige_erosion))
#cv_show('res',res)
pie= cv2.imread('./pie.png')
# cv_show('pie',pie)
kernel = np.ones((30,30),np.uint8)
dilate_1 = cv2.dilate(pie,kernel,iterations = 1)
dilate_2 = cv2.dilate(pie,kernel,iterations = 2)
dilate_3 = cv2.dilate(pie,kernel,iterations = 3)
res = np.hstack((dilate_1,dilate_2,dilate_3))
# cv_show('res',res)
#============================开运算与闭运算============================
# 开：先腐蚀，再膨胀
img = cv2.imread('dige.png')

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
res = np.hstack((img,opening,closing))
# cv_show('res',res)
#============================梯度运算============================
pie = cv2.imread('./pie.png')
kernel = np.ones((7,7),np.uint8)
dilate = cv2.dilate(pie,kernel,iterations=5)
erode = cv2.dilate(pie,kernel,iterations=5)
res = np.hstack((pie,dilate,erode))
#cv_show('res',res )
#梯度计算
gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)
# cv_show('gradient',gradient)
#============================礼帽与黑帽============================
img = cv2.imread('./dige.png')
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel)
res = np.hstack((tophat,blackhat))
# cv_show('res',res)
#============================图像梯度-Sobel算子============================
img = cv2.imread('./lena.jpg',cv2.IMREAD_GRAYSCALE)#读如灰色图像
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
res = np.hstack((img,sobelxy))
# cv_show('res',res)
#============================图像梯度-Scharr算子============================
scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
#============================图像梯度-laplacian算子============================
laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
res = np.hstack((img,sobelxy,scharrxy,laplacian))
# cv_show('res',res)
#============================Canny边缘检测============================
img=cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)
v1=cv2.Canny(img,80,150)
v2=cv2.Canny(img,50,100)#设置的是双边阈值
res = np.hstack((img,v1,v2))
# cv_show('res',res)
#============================图像金字塔============================
            #============================高斯金字塔============================
img = cv2.imread('./AM.png')
print(img.shape)
up=cv2.pyrUp(img)
#cv_show('up',up)#只是将图片的大小改变了,并且是2倍2倍的增长。
print (up.shape)
down=cv2.pyrDown(img)
#cv_show('down',down)
print(down.shape)
up_down=cv2.pyrDown(up)
# cv_show('up_down',np.hstack((img,up_down)))
            #============================拉普拉斯金字塔============================
down=cv2.pyrDown(img)
down_up=cv2.pyrUp(down)
l_1=img-down_up
# cv_show('l1',l_1)
#============================图像轮廓============================
img = cv2.imread('contours.png')
# cv_show('img',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv_show('thresh',thresh)
#后面的两个参数一个是 轮廓检索模式，另一个是轮廓逼近方法
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv_show('img',img)
#
draw_img = img.copy()
'''
第二个参数是轮廓，一个python列表，
第三个参数是轮廓的索引（在绘制独立轮廓是很有用，当设置为-1时绘制所有轮廓）。
接下来的参数是轮廓的颜色和厚度。
'''
#绘制轮廓
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
draw_img = img.copy()
print(len(contours))
res = cv2.drawContours(draw_img, contours, 2, (0, 0, 255), 2)
# cv_show('res',res)
    #轮廓特征
cnt = contours[0]
        #面积
print(cv2.contourArea(cnt))
        #周长，True表示闭合的
print(cv2.arcLength(cnt,True))
    #轮廓近似
img = cv2.imread('./contours2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt0 = contours[0] #这个是一个里面的线
cnt1 = contours[1] #这个是一个外边的线 contours是全部的线
# print(len(contours))
draw_img = img.copy()
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
res = cv2.drawContours(draw_img,[cnt0],-1,(0,0,255),2)
res = cv2.drawContours(draw_img,[cnt1],-1,(0,0,255),2)
# cv_show('res',res)
    #边界矩形
img = cv2.imread('contours.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# cv_show('img',img)
    #外接圆
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),1)
# cv_show('img',img)
# cv_show('res',res)

#============================傅里叶变换============================
img = cv2.imread('lena.jpg')
img_float32 = np.float32(img)


#============================图像融合============================
#============================图像融合============================
#============================图像融合============================
#============================图像融合============================
#============================图像融合============================
#============================图像融合============================
