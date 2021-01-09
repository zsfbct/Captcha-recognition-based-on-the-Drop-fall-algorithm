import cv2
import numpy as np
import os

for fileName in os.listdir('E:/testPic/pic/VCR_train_data'):
    img_path = 'E:\\testPic\\pic\\VCR_train_data\\' + fileName
    img = cv2.imdecode(np.fromfile(img_path, dtype = np.uint8), -1)

    #灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #动态阈值二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 1)

    #高斯模糊，把断掉的字符连接起来
    blur = cv2.GaussianBlur(binary, (11,11), 1.5)

    #重新二值化
    ret, re_img = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    height, width = re_img.shape

    #去掉干扰线
    scanSize = [3, 10]
    position = [1,0]
    step = 5
    while position[0] + scanSize[0] <height:
        while position[1] + scanSize[1] < width:
            #数一数这个筛选框内有多少个黑格子
            count = 0
            for i in range(scanSize[0]):
                for j in range(scanSize[1]):
                    count += re_img[position[0] + i][position[1] + j]
            count = scanSize[0] * scanSize[1] - count/255
            #如果80%的像素都是黑的，那么大概率是干扰线了
            if count > scanSize[0] * scanSize[1] * 0.6:
                #如果在某个位置，上下都是白像素，那么中间的像素全部设为白
                for j in range(scanSize[1]):
                    if re_img[position[0] - 1][position[1] + j] == 255 and re_img[position[0] + scanSize[0]][position[1] + j] == 255:
                        for i in range(scanSize[0]):
                            re_img[position[0] + i][position[1] + j] = 255
            position[1] += step
        position[0] += scanSize[0]
        position[1] = 0

    #黑白颠倒60*160
    for i in range(height):
        for j in range(width):
            re_img[i, j] = 255 - re_img[i, j]

    #把图片划分连通域
    contours, hierarchy = cv2.findContours(re_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #根据帕累托法则，选择面积之和超过60%的几个最大的连通域保留，其余的删除
    areas = []
    key = 0
    for cntr in contours:
        areas.append([cv2.contourArea(cntr), key])
        key = key + 1

    for i in range(1, len(areas)):
        flag = False
        for j in range(len(areas) - i):
            if areas[j+1][0] < areas[j][0]:
                tmp = areas[j]
                areas[j] = areas[j+1]
                areas[j+1] = tmp
                flag = True
        if flag == False:
            break

    partialSum = 0
    total = sum(item[0] for item in areas)

    cntrDelete = []
    for item in areas:
        if (partialSum + item[0]) < 0.4 * total:
            cntrDelete.append(contours[item[1]])
            partialSum += item[0]
        else:
            break

    #不需要的涂黑
    cv2.fillPoly(re_img, cntrDelete, (0, 0, 0))

    #再一次黑白颠倒
    for i in range(height):
        for j in range(width):
            re_img[i, j] = 255 - re_img[i, j]     

    #cv2.imshow('Image', re_img)

    img_path = 'E:\\single_train_2\\' + fileName
    cv2.imwrite(img_path, re_img)
