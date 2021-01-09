'''
用于提取图片中的连通域
'''
import cv2 as cv
import numpy as np
from functools import lru_cache
import pickle

# 载入已经训练好的SVM模型，预测连通域中的字符数量
with open('.contour-classifier', 'rb') as file:
    contour_clasifier = pickle.load(file)

with open('.contour-classifier-preprocessor', 'rb') as file:
    contour_classifier_preprocessor = pickle.load(file)


class Contour:
    '''
    Represents a contour extracted from an image.
    '''
    def __init__(self, points, img, children=[]):
        '''
        Initializes this instance
        :param points: Must be a list the list of points that defines the contour
        See OpenCV contours
        :param children: A list of optional inner contours
        '''
        self.points = points
        self.img = img
        self.children = []


    @property
    @lru_cache(maxsize=1)
    def bbox(self):
        '''
        Returns a non rotated rectangle that encapsulates this contour
        '''
        return ContourBBox(self)

    @property
    def bbox_width(self):
        '''
        Its the same as contour.bbox.width
        '''
        return self.bbox.width

    @property
    def bbox_height(self):
        '''
        Its the same as contour.bbox.height
        '''
        return self.bbox.height

    @property
    def bbox_area(self):
        '''
        Its the same as contour.bbox.area
        '''
        return self.bbox.area

    @property
    def bbox_ratio(self):
        '''
        Its the same as contour.bbox.ratio
        '''
        return self.bbox.ratio

    @property
    def area(self):
        '''
        Returns the area covered by this contour
        '''
        return cv.contourArea(self.points)

    @property
    def extent(self):
        '''
        Returns the ratio between the areas of the contour and its bounding rectangle
        '''
        return self.area / self.bbox_area

    @property
    def perimeter(self):
        '''
        Returns the perimeter of this contour
        '''
        return cv.arcLength(self.points, True)

    @property
    def bbox_mask(self):
        '''
        Its an alias of self.bbox.mask
        '''
        return self.bbox.mask

    def draw(self, img, show_children=False):
        '''
        Draws this contour over the image specified.  Must be an RGB image with uint8 data type
        :param show_children: When this is set to True, it also draws children contours
        '''
        if show_children:
            for child in self.children:
                img = child.draw(img, show_children)
        return self._draw(img)


    def draw_bbox(self, img, show_children=False):
        '''
        Its the same as self.bbox.draw(). Also if show_children is True, it calls
        draw_bbox() on children contours
        '''
        if show_children:
            for child in self.children:
                img = child.draw_bbox(img, show_children)
        return self.bbox.draw(img)

    def extract_bbox_pixels(self, img):
        '''
        Its an alias of self.bbox.extract_pixels
        '''
        return self.bbox.extract_pixels(img)


    @property
    @lru_cache(maxsize=1)
    def num_chars(self):
        '''
        :return Returns an intenger number in the range [0, 5] indicating the
        predicted number of chars inside this contour.
        '''
        return self.num_chars_proba.argmax().item()

    @property
    def num_chars_proba(self):
        '''
        Returns an array of size m where the element at index k is the predicted probability
        of this contour of having k characters inside it
        All elements are in the interval [0, 1] and their sum gives 1
        The probability of this contour of having more than m characters is considered 0
        '''
        model = contour_clasifier
        scaler = contour_classifier_preprocessor

        X = np.array([
            self.bbox_width, self.bbox_height,
            self.area, self.extent, self.perimeter
        ], dtype=np.float32).reshape([1, -1])

        X = scaler.transform(X)

        y = model.predict_proba(X)[0]
        return y

    @property
    @lru_cache(maxsize=1)
    def properties(self):
        '''
        Returns a dictionary carrying information about this contour
        '''
        return {
            'bbox_width': self.bbox_width,
            'bbox_height': self.bbox_height,
            'area': self.area,
            'extent': self.extent,
            'perimeter': self.perimeter
        }


    def _draw(img, color=(0, 255, 0)):
        return cv.drawContours(img, self.points, -1, tuple(color))

class ContourBBox:
    '''
    Represents an AABB Rectangle bounding box of a contour
    '''
    def __init__(self, contour):
        '''
        :param contour: Is the contour that is surrounded by this rectangle box
        '''
        self.inner_contour = contour
        self.left, self.top, self.width, self.height = cv.boundingRect(contour.points)


    @property
    def right(self):
        '''
        Returns the right side of the rectangle
        '''
        return self.left + self.width

    @property
    def bottom(self):
        '''
        Returns the bottom of the rectangle
        '''
        return self.top + self.height

    @property
    def area(self):
        '''
        Returns the area of the rectangle
        '''
        return self.width * self.height

    @property
    def ratio(self):
        '''
        Returns the ratio: width/height of the rectangle
        '''
        return self.width / self.height

    @property
    @lru_cache(maxsize=1)
    def mask(self):
        '''
        Returns an image of the same size as this bounding box where pixels are
        painted with white color if falls inside the contour
        '''
        return self.extract_pixels(self.inner_contour.img)

    def extract_pixels(self, img):
        '''
        Returns an image of the same size as this bounding box extracting the pixels
        from the image
        '''
        return img[self.top:self.top+self.height, self.left:self.left+self.width]


    def draw(self, img, color=(0, 255, 0)):
        '''
        Draws this rectangle over the specified image. Must be an RGB image with uint8 data type
        '''
        img = cv.rectangle(img,
                        (self.left,self.top), (self.left+self.width,self.top+self.height),
                        tuple(color), 1)
        return img

    def __str__(self):
        return '({}, {}), {} x {}'.format(self.left, self.top, self.width, self.height)


'''提取图片中的连通域'''
def find_contours(img0):
    '''
    输入的img0是【0，255】之间的 通过imread（file，0）读取的
    '''
    img = (255 - img0).astype(np.uint8)
    blurred = cv.medianBlur(img, 3)

    # Detect edges
    edges = cv.Laplacian(img, 5).clip(0, 255).astype(np.uint8)

    # Find contours
    contours, hierachy = cv.findContours(blurred, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    nn = len(contours)
    contour_parent = hierachy[0, :, 3]
    contour_children = [set() for k in range(0, nn)]

    for k in range(0, nn):
        parent = contour_parent[k]
        if parent != -1:
            contour_children[parent].add(k)

    # Return all the contours (instances of class Contour)

    items = {}
    A = set([k for k in range(0, nn) if len(contour_children[k]) == 0])
    B = set(range(0, nn)) - A
    for k in A:
        items[k] = Contour(contours[k], img)

    while len(B) > 0:
        C = B & set([contour_parent[k] for k in A])
        for j in C:
            items[j] = Contour(contours[j], img, [items[k] for k in contour_children[j]])
        B -= C

    return [items[k] for k in range(0, nn) if contour_parent[k] == -1]


def draw_contours(img, contours, show_children=False):
    '''
    Draws all the contours over the specified image (must be a 2D array
    with float32 values with the intensities of the pixels in the range [0, 1])
    :param show_children: If this is disabled, only top level contours are drawn
    '''
    for contour in contours:
        img = contour.draw(img, show_children)
    return img


def draw_bbox_contours(img, contours, show_children=False):
    '''
    Draw all the bounding boxes of the contours indicated over the specified image
    :param show_children: if thisis disable, only top level contours are drawn
    '''
    for contour in contours:
        img = contour.draw_bbox(img, show_children)
    return img


