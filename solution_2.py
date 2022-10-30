import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def findRotationAndTranslation(img_1, img_2):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    keyPoints_1, descriptors_1 = sift.detectAndCompute(img_1, None)
    keyPoints_2, descriptors_2 = sift.detectAndCompute(img_2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k = 2)

    points_1 = []
    points_2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            points_2.append(keyPoints_2[m.trainIdx].pt)
            points_1.append(keyPoints_1[m.queryIdx].pt)

    points_1 = np.int32(points_1)
    points_2 = np.int32(points_2)
    Essential_matrix, mask = cv.findEssentialMat(points_1, points_2, focal=100.0, pp=(960, 540), method=cv.RANSAC, prob=0.999, threshold=1.0)

    _, RotationMatrix, translation, mask = cv.recoverPose(Essential_matrix, points_1, points_2, focal=100, pp=(960, 540))
    return RotationMatrix, translation

img_1 = cv.imread('img1.png')
img_2 = cv.imread('img2.png')
img_3 = cv.imread('img3.png')
img_1_gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
img_2_gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
img_3_gray = cv.cvtColor(img_3, cv.COLOR_BGR2GRAY)

#call the findRotationAndTranslation function
RotationMatrix_1_2, translation_1_2 = findRotationAndTranslation(img_1_gray, img_2_gray)
RotationMatrix_1_3, translation_1_3 = findRotationAndTranslation(img_1_gray, img_3_gray)

print(" 6 DOF pose estimation between img_1 and img_2")
print("Rotation matrix:")
print(RotationMatrix_1_2)
print("Translation vector:")
print(translation_1_2)

print(" 6 DOF pose estimation between img_1 and img_3")
print("Rotation matrix:")
print(RotationMatrix_1_3)
print("Translation vector:")
print(translation_1_3)