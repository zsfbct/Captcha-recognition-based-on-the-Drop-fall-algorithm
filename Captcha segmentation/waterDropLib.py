'''
滴水算法实现
'''
from skimage import morphology, draw
import numpy as np



def getCast(skeleton):
    cast = []
    for column in range(skeleton.shape[1]):
        count = 0
        for row in range(skeleton.shape[0]):
            if skeleton[row][column] == True:
                count = count + 1
        cast.append(count)
    return cast


def getJunction(skeleton, leftBorder, rightBorder):
    junction = []
    #定义一个移动集合
    move = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]
    for column in range(max([int(leftBorder), 1]), min([int(rightBorder), skeleton.shape[1] - 1])):
        for row in range(1, skeleton.shape[0] - 1):
            if skeleton[row][column] == True:
                #绕一圈，计算有多少次像素的变换
                connection = 0
                for i in range(8):
                    if skeleton[row+move[i][0]][column+move[i][1]] == True and skeleton[row+move[i+1][0]][column+move[i+1][1]] == False:
                        connection = connection + 1
                if connection > 2:
                    junction.append([row, column])
    return junction


def getDropRoute(img, skeleton, index):
    tolerance = 15
    #获取骨架的交叉点
    junction = getJunction(skeleton, index - tolerance, index + tolerance)
    seperateLine = []
    seperateLine.append([0, index])
    #滴落开始
    for i in range(skeleton.shape[0] - 1):
        if img[seperateLine[-1][0]+1][seperateLine[-1][1]] == 0:
            #如果水滴下方没有字符，那么直接下落
            seperateLine.append([seperateLine[-1][0]+1, seperateLine[-1][1]])
        elif img[seperateLine[-1][0]+1][seperateLine[-1][1]-1] == 0 and seperateLine[-1][1]-1 >= index - tolerance:
            #如果水滴左下方没有字符，那么向左下方下落
            seperateLine.append([seperateLine[-1][0]+1, seperateLine[-1][1]-1])
        elif img[seperateLine[-1][0]+1][seperateLine[-1][1]+1] == 0 and seperateLine[-1][1]+1 <= index + tolerance:
            #如果水滴右下方没有字符，那么向右下方下落
            seperateLine.append([seperateLine[-1][0]+1, seperateLine[-1][1]+1])
        else:
            #如果水滴周围都有字符，那么尽量跟着骨架的交叉点走
            #寻找在水滴下方的，横向距离不是很远的最近交叉点
            flag = False
            for point in junction:
                if point[0] > seperateLine[-1][0] + 1 and abs(point[1] - seperateLine[-1][1]) <= tolerance and point[1] - seperateLine[-1][1] > 0:
                    gradient = (seperateLine[-1][0] - point[0])/(seperateLine[-1][1] - point[1])
                    seperateLine.append([seperateLine[-1][0]+1, seperateLine[-1][1]+int(1/gradient)])
                    flag = True
                    break
            #如果下方没有交叉点了，那么直接下落吧
            if flag == False:
                seperateLine.append([seperateLine[-1][0]+1, seperateLine[-1][1]])
    return seperateLine


def dropFall(img, initialPosition, num_chars):
    # 图片和推荐的起始分割位置（列坐标列表）
    # 提取骨架
    re_img = img
    skeleton = morphology.skeletonize(re_img)
    # 投影判断一共有几个字符块
    cast = getCast(img)
    partIndex = []
    if cast[0] != 0:
        partIndex.append(0)
    for row in range(1, len(cast)):
        if cast[row - 1] == 0 and cast[row] != 0:
            partIndex.append(row)
        elif cast[row - 1] != 0 and cast[row] == 0:
            partIndex.append(row)
    if cast[-1] != 0:
        partIndex.append(len(cast) - 1)

    pic = np.ones([num_chars, skeleton.shape[0], skeleton.shape[1]], dtype=None, order='C')
    # 整合分割点和字符块起始点
    for position in initialPosition:
        partIndex.append(position)
        partIndex.append(position)
    partIndex.sort()

    # 判断每一个地方是否需要滴水分割
    for i in range(num_chars):
        #         if partIndex[i*2] not in initialPosition and partIndex[i*2+1] not in initialPosition:
        #             #此处为独立的字符
        #             for row in range(skeleton.shape[0]):
        #                 for column in range(partIndex[i*2], partIndex[i*2+1] + 1):
        #                     pic[i][row][column] = 1 - img[row][column]
        if partIndex[i * 2] not in initialPosition and partIndex[i * 2 + 1] in initialPosition:
            # 此处前面不需要分割，后面需要分割
            dropRoute = getDropRoute(img, skeleton, partIndex[i * 2 + 1])
            while len(dropRoute) > 0:
                temp = dropRoute.pop()
                for column in range(partIndex[i * 2], temp[1] + 1):
                    pic[i][temp[0]][column] = 1 - img[temp[0]][column]
        elif partIndex[i * 2] in initialPosition and partIndex[i * 2 + 1] not in initialPosition:
            # 此处前面需要分割，后面不需要分割
            dropRoute = getDropRoute(img, skeleton, partIndex[i * 2])
            while len(dropRoute) > 0:
                temp = dropRoute.pop()
                for column in range(temp[1], partIndex[i * 2 + 1] + 1):
                    pic[i][temp[0]][column] = 1 - img[temp[0]][column]
        else:
            # 此处前后都需要分割
            dropRoute1 = getDropRoute(img, skeleton, partIndex[i * 2])
            dropRoute2 = getDropRoute(img, skeleton, partIndex[i * 2 + 1])
            while len(dropRoute1) > 0:
                temp1 = dropRoute1.pop()
                temp2 = dropRoute2.pop()
                for column in range(temp1[1], temp2[1] + 1):
                    pic[i][temp1[0]][column] = 1 - img[temp1[0]][column]
    # 返回一个4*60*160的数组，每个60*160各自包含一个切割后的字符，字符是0，背景是1
    return pic
