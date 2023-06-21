import cv2
import numpy as np
import matplotlib.pyplot as plt

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#=================================直方图=================================
img = cv2.imread('cat.jpg',0)#0表示灰度图
#分别代表的是：图像，灰度图，设置None，高度为56，宽度为0-256
hist = cv2.calcHist([img],[0],None,[256],[0,256])
print(hist.shape)
print(img.shape)
# plt.hist(img.ravel(),256)#统计次数使用的
# plt.show()
img = cv2.imread('cat.jpg')
color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[250],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()

#============mask操作============
maskmask = np.zeros(img.shape[:2],np.uint8)
print(maskmask.shape)
print(img.shape[:2])
maskmask[100:300,100:400] = 255#中间区域设置为全白色
# cv2.imshow('mask',maskmask)
img = cv2.imread('cat.jpg',0)#读成灰色图片
# cv2.imshow('img',img)
masked_img = cv2.bitwise_and(img,img,mask=maskmask)
# cv2.imshow('masked_img',masked_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],maskmask,[256],[0,256])

#============直方图均衡化============
img = cv2.imread('clahe.jpg',0)
# plt.hist(img.ravel(),256)
# plt.show()
#旨在使得图像整体效果均匀，黑与白之间的各个像素级之间的点更均匀一点。
equ = cv2.equalizeHist(img)
# plt.hist(equ.ravel(),256)
# plt.show()
#进行对比，均值化之后的，没有均值化之后的。
res = np.hstack((img,equ))
#cv_show('res',res)
#============自适应直方图均衡化============
#clipLimit：颜色对比度的阈值，可选项，默认值 8
#titleGridSize：局部直方图均衡化的模板（邻域）大小，可选项，默认值 (8,8)
clache = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
res_clahe = clache.apply(img)
res = np.hstack((img,equ,res_clahe))#进行水平叠加
#cv_show('res',res)
#=================================模板匹配=================================
img = cv2.imread('lena.jpg',0)
template = cv2.imread('face.jpg',0)
h,w, = template.shape
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF)
#找出矩阵中最大值和最小值，即其对应的(x, y)的位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
for meth in methods:
    img2 = img.copy()

    #匹配方法的真值
    method = eval(meth)
    print(method)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)#获取到这个矩阵中最小值、最大值、最小值索引、最大值索引

    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right =  (top_left[0]+w,top_left[1]+h)

    #画矩形
    # cv2.rectangle(img2,top_left,bottom_right,255,2)
    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    # plt.subplot(122), plt.imshow(img2, cmap='gray')
    # plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.show()
#=================================匹配多个对象=================================
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)#返回的是索引
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)
#=================================模板匹配=================================
