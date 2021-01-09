'''
进行连通域中的字符分割
'''

from functools import partial
from itertools import product, repeat, permutations, combinations_with_replacement, chain
from waterDropLib import *

import numpy as np
import cv2 as cv
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from contours import find_contours


'''所有的find_chars函数都是输入[0,255] 返回[0,1].float32'''

'''找到初始分割点，输出包含字符数量大于1的连通域图片信息以及所包含的字符数量，返回分割线坐标'''
def find_char_separators(img, num_chars):

    f = (img > 0).mean(axis=0)
    a, b = f.min(), f.max()
    f = (f - a) / (b - a)
    n, k = len(f), num_chars
    x = np.arange(0, n)

    # Initial guess
    s0 = np.linspace(0, n, k+1)[1:-1]

    # Value boundaries for each delimiter.
    char_min_size = 15
    delimiter_margin = 8
    bounds = np.transpose(np.stack([
                            np.maximum(s0 - delimiter_margin, 0),
                            np.minimum(s0 + delimiter_margin, n-1)],
                        axis=0))

    # Now we move the delimiters to divide the chars better

    y_spl = UnivariateSpline(x, f, s=0,k=4)
    y_spl_df = y_spl.derivative(n=1)

    F = lambda s: np.sum(y_spl(s))
    dF = lambda s: y_spl_df(s) / (k - 1)
    jac = lambda s, *args: dF(s)

    result = minimize(F, s0, jac=jac, method='SLSQP', bounds=bounds)
    s = np.round(result.x)

    separators = s.astype(np.uint16)
    return separators

'''将切割好的字符尺寸调整到60*40，输入为字符图片信息，输出为60*40的字符图片'''
def process_image(img, dsize):

    res = cv.resize(img, (40,60), interpolation=cv.INTER_AREA)
    ret, thresholded = cv.threshold(res*255, 140, 255, cv.THRESH_BINARY)
    thresholded=(thresholded/255).astype(np.float32)

    return thresholded

'''进行竖直切割，输入为分割点坐标，输出为切割后的字符图片信息'''
def split_array(a, separators, axis=1):

    seperators = sorted(separators)
    n_sep = len(separators)

    if n_sep == 1:
        sep = separators[0]
        a = a.swapaxes(0, axis)
        return [a[0:sep].swapaxes(0, axis), a[sep:].swapaxes(0, axis)]

    head, body = split_array(a, [separators[0]], axis)
    splits = split_array(body, np.array(separators[1:]) - separators[0], axis)
    return [head] + splits

'''提取字符函数 用于提取滴水算法切割之后的图片中文字'''
def extract_contour(pic):
    extract = []
    for i in range(len(pic)):
        col = (pic[i] == 0).sum(axis=0)  ##(54)按照列相加
        row = (pic[i] == 0).sum(axis=1)  ##(60)按照行相加
        # 寻找左右边界
        left = 0
        right = pic[i].shape[0]
        for m in range(0, pic[i].shape[1]):
            if col[m] != 0:
                left = max(0, m - 5)
                break
        for m in range(pic[i].shape[1] - 1, -1, -1):
            if col[m] != 0:
                right = min(m + 5, pic[i].shape[1] - 1)
                break
        # 寻找上下边界
        up = 0
        down = pic[i].shape[1]
        for m in range(0, pic[i].shape[0]):
            if row[m] != 0:
                up = max(0, m - 5)
                break
        for m in range(pic[i].shape[0] - 1, -1, -1):
            if row[m] != 0:
                down = min(m + 5, pic[i].shape[0] - 1)
                break
        #         extract[i]=process_image(pic[i][up:(down+1),left:(right+1)],(60,40))
        extract.append(pic[i][up:(down + 1), left:(right + 1)])

    return extract

'''分割方法一：直接在初始分割点处进行竖直切割'''
def find_chars(img, char_size, num_chars):
    '''
    img[0,255] 通过imread(file,0)读入
    '''
    num_chars = 4

    # Extract image contours
    contours0 = find_contours(img)
    contours = []

    # Remove contours which we predict that dont have any char inside
    contours = [contour for contour in contours0 if contour.num_chars > 0]
    assert len(contours) > 0

    k = len(contours)

    # Sort frames by its horizontal position (from left to right)
    contours.sort(key=lambda contour: contour.bbox.left)

    # Now we create a 2D matrix where the element at index i,j
    # will be the probability of the frame i to contain j characters inside
    P = np.array([contour.num_chars_proba for contour in contours])

    # If n0, n1, ..., nk are the number of predicted characters inside each frame, we find the best configuration so that n0 + n1 + ... + nk = num_chars
    # and ensure that P[0, n[0]] * P[1, n[1]] * ... * P[k, n[k]] is maximized

    # All valid configurations (n0, n1, ..., nk) such that n0 + n1 + ... + nk = num_chars
    configs = filter(lambda x: np.sum(x) == num_chars, combinations_with_replacement(range(0, num_chars + 1), k))
    configs = list(frozenset(chain.from_iterable(map(lambda config: permutations(config, k), configs))))

    configs = np.array(configs, dtype=np.uint8)
    nc = configs.shape[0]

    # Calculate a score function for each configuration
    scores = np.zeros([nc]).astype(np.float32)

    for i in range(0, nc):
        scores[i] = np.prod(P[np.arange(0, k), configs[i]])

    # Get the best configuration
    best_config = configs[np.argmax(scores)]

    # Split the contours into frames
    img = (img).astype(np.uint8)
    frames = []

    for k in range(0, k):
        if best_config[k] == 0:
            continue

        elif best_config[k] == 1:
            # Contour boundaries only holds 1 char.
            frame = contours[k].extract_bbox_pixels(img)
            frames.append(frame/255)
        else:
            # Contour holds more than 1 char
            # Split it into multiple frames
            separators = find_char_separators(contours[k].bbox_mask, best_config[k])
            splits = split_array(contours[k].extract_bbox_pixels(img), separators, axis=1)
            for i in splits:
                frames.append(i/255)

    for i, frame in enumerate(frames):
        if frame.shape[1] < 1:
            frames[i] = np.zeros([40, 60])

    processed_frames = map(partial(process_image, dsize=char_size), frames)
    chars = np.stack([(frame.astype(np.float16)) for frame in processed_frames], axis=0)
    chars = chars.astype(np.uint8)

    return chars

'''分割方法二：以初始分割点为水滴落点进行滴水算法切割'''
def find_chars_water_extract(img, char_size, num_chars):
    char_size = (40, 60)
    '''
    img[0,255] 通过imread(file,0)读入
    '''
    num_chars = 4

    # Extract image contours
    contours0 = find_contours(img)
    contours = []

    # Remove contours which we predict that dont have any char inside
    contours = [contour for contour in contours0 if contour.num_chars > 0]
    assert len(contours) > 0

    k = len(contours)

    # Sort frames by its horizontal position (from left to right)
    contours.sort(key=lambda contour: contour.bbox.left)

    # Now we create a 2D matrix where the element at index i,j
    # will be the probability of the frame i to contain j characters inside
    P = np.array([contour.num_chars_proba for contour in contours])

    # If n0, n1, ..., nk are the number of predicted characters inside each frame, we find the best configuration so that n0 + n1 + ... + nk = num_chars
    # and ensure that P[0, n[0]] * P[1, n[1]] * ... * P[k, n[k]] is maximized

    # All valid configurations (n0, n1, ..., nk) such that n0 + n1 + ... + nk = num_chars
    configs = filter(lambda x: np.sum(x) == num_chars, combinations_with_replacement(range(0, num_chars + 1), k))
    configs = list(frozenset(chain.from_iterable(map(lambda config: permutations(config, k), configs))))

    configs = np.array(configs, dtype=np.uint8)
    nc = configs.shape[0]

    # Calculate a score function for each configuration
    scores = np.zeros([nc]).astype(np.float32)

    for i in range(0, nc):
        scores[i] = np.prod(P[np.arange(0, k), configs[i]])

    # Get the best configuration
    best_config = configs[np.argmax(scores)]

    # Split the contours into frames
    img = (img).astype(np.uint8)
    frames = []

    for k in range(0, k):
        if best_config[k] == 0:
            continue

        elif best_config[k] == 1:
            # Contour boundaries only holds 1 char.
            frame = contours[k].extract_bbox_pixels(img)
            frames.append(frame / 255)
        else:
            # Contour holds more than 1 char
            # Split it into multiple frames
            separators = find_char_separators(contours[k].bbox_mask, best_config[k])
            ### 在这里应用水滴算法###
            con = 1 - contours[k].extract_bbox_pixels(img) / 255
            pic = dropFall(con, separators, best_config[k])
            extract = extract_contour(pic)
            frames.extend(extract)

    processed_frames = map(partial(process_image, dsize=char_size), frames)
    chars = np.stack([(frame.astype(np.float16)) for frame in processed_frames], axis=0)
    chars = chars.astype(np.uint8)

    return chars


