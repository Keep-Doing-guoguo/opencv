import cv2
import matplotlib.pyplot as plt
import numpy as py

#============================数据读取-图像============================
img  = cv2.imread('./cat.jpg')
# print(img)
# print(img.shape)
#显示图像，也可以创建多个窗口
# cv2.imshow('cat',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)#等待时间，毫秒级，0表示任意键中指
    cv2.destroyAllWindows()
img = cv2.imread('./cat.jpg',cv2.IMREAD_GRAYSCALE)#默认是彩色图像，现在设置为灰色图像。
# cv_show('cat',img)#显示图像的
# print(img)
# print(img.shape)
#可以使用保存
# cv2.imwrite()
#============================数据读取-视频============================
# vc = cv2.VideoCapture(0)
# #检查是否打开正确
# if vc.isOpened():
#     oepn,frame = vc.read()
# else:
#     oepn = False
# while oepn:
#     ret,frame = vc.read()
#     if frame is None:
#         break
#     if ret == True:
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         cv2.imshow('result',gray)
#         if cv2.waitKey(10) & 0xFF == 27:
#             break
# vc.release()
# cv2.destroyAllWindows()


#============================截取部分图像数据============================
img = cv2.imread('./cat.jpg')
# cat = img[0:50,0:200]
# cv_show('cat',cat)

#============================颜色通道提取============================
b,g,r=cv2.split(img)
print(r.shape)
img = cv2.merge((b,g,r))
print(img.shape)
# 只保留R
#opencv读取的图片是BGR的格式，第一维度代表的是B，第二维度代表的是G，第三维度代表的是R，所以保留最后一个维度，只需要将前面两个设置为0即可。
copy_img = img.copy()
copy_img[:,:,0] = 0
copy_img[:,:,1] = 0
# cv_show('R',copy_img)

copy_img = img.copy()
copy_img[:,:,0] = 0
copy_img[:,:,2] = 0
# cv_show('R',copy_img)

#============================边界填充============================
top_size,bottom_size,left_size,right_size = (50,50,50,50)
#总共有5种填充方法,填充函数使用的是copyMakeBorder，方法可以选择cv2.,img是图片，然后设置填充大小
'''
BORDER_REPLICATE：复制法，也就是复制最边缘像素。
BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
BORDER_CONSTANT：常量法，常数值填充。
'''
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
# fig,ax = plt.subplots(2,3,figsize=(20,6))
# pic = [img,replicate,reflect,reflect101,wrap,constant]
# ax[0][0].imshow(pic[0])
#
# ax[0][1].imshow(pic[1])
#
# ax[0][2].imshow(pic[2])
#
# ax[1][0].imshow(pic[3])
#
# ax[1][1].imshow(pic[4])
#
# ax[1][2].imshow(pic[5])
# plt.show()
#============================数值计算============================
img_cat = cv2.imread('./cat.jpg')
img_dog = cv2.imread('./dog.jpg')
img_cat2=img_cat+10
#============================图像融合============================
# img_cat + img_dog  #shapes (414,500,3) (429,499,3)
img_dog = cv2.resize(img_dog,(500,414))#resize函数，不考虑图像形变问题。
print(img_dog.shape)
#两张图片的权重相同，gamma修正系数,0为亮度值
res = cv2.addWeighted(img_cat,0.5,img_dog,0.5,0)#主要功能就是将两幅图像合成为一幅图像
# plt.imshow(res)
# plt.show()

res = cv2.resize(img,(0,0),fx=4,fy=4)#x轴y轴的倍数
plt.imshow(img)
plt.show()