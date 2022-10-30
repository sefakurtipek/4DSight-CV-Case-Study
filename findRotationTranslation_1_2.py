import numpy as np
import cv2 as cv
import cv2 as cv2
from matplotlib import pyplot as plt
from scipy.linalg import svd

#if we know the intrinsic camera parameter matrix we can convert the image points to normalized image coordinates
#origin is in the center of the image
#effective focal lenght equals to 1

cx = 960
cy = 540
f = 100
# K is the intrinsic matrix of the camera
K = np.array([[f, 0, cx],
             [0, f, cy],
             [0, 0,  1]], dtype = "double")

# known 2D and 3D points. There are 20 points
correspondance_2D = np.load('vr2d.npy')
correspondance_3D = np.load('vr3d.npy')

img_1 = cv.imread('img1.png')
img_2 = cv.imread('img2.png')
img_3 = cv.imread('img3.png')
img_1_gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
img_2_gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
img_3_gray = cv.cvtColor(img_3, cv.COLOR_BGR2GRAY)

#############################################################################
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
keyPoints_1, descriptors_1 = sift.detectAndCompute(img_1, None)
keyPoints_2, descriptors_2 = sift.detectAndCompute(img_2, None)

# Marking the keypoint on the image using circles
img = cv.drawKeypoints(img_1_gray, keyPoints_1 , img_1_gray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Keypoints1_in_img1.png', img)

# Marking the keypoint on the image using circles
img = cv.drawKeypoints(img_2_gray, keyPoints_2, img_2_gray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Keypoints2_in_img2.png', img)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_1, descriptors_2, k = 2)

# Apply ratio test
good = []
good_without_list = []
for m,n in matches:
    if m.distance < 0.18*n.distance:
        good.append([m])
        good_without_list.append(m)
print('good_without_list: ',len(good_without_list))

#############################################################################
MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keyPoints_1[m.queryIdx].pt for m in good_without_list ]).reshape(-1,1,2)
    dst_pts = np.float32([ keyPoints_2[m.trainIdx].pt for m in good_without_list ]).reshape(-1,1,2)

src_pts = src_pts.reshape((src_pts.shape[0], 2))

one = np.ones((len(good_without_list),1))
src_pts = np.append(src_pts, one, axis=1) #making image points as homogeneous points

dst_pts = dst_pts.reshape((dst_pts.shape[0], 2))
dst_pts = np.append(dst_pts, one, axis=1) #making image points as homogeneous points

def compute_fundamental_matrix(points1, points2):
    '''
    Compute the fundamental matrix given the point correspondences
    
    Parameters
    ------------
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as 
        homogeneous coordinates
    '''
    # validate points
    assert points1.shape[0] == points2.shape[0], "no. of points don't match"
    
    u1 = points1[:, 0]
    v1 = points1[:, 1]
    u2 = points2[:, 0]
    v2 = points2[:, 1]
    one = np.ones_like(u1)
    
    # construct the matrix 
    # A = [u2.u1, u2.v1, u2, v2.u1, v2.v1, v2, u1, v1, 1] for all the points
    # stack columns
    A = np.c_[u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, one]
    
    # peform svd on A and find the minimum value of |Af|
    U, S, V = np.linalg.svd(A, full_matrices=True)
    f = V[-1, :]
    F = f.reshape(3, 3) # reshape f as a matrix
    
    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F, full_matrices=True)
    S[-1] = 0 # zero out the last singular value
    F = U @ np.diag(S) @ V
    return F

def compute_fundamental_matrix_normalized(points1, points2):
    '''
    Normalize points by calculating the centroid, subtracting 
    it from the points and scaling the points such that the distance 
    from the origin is sqrt(2)
    
    Parameters
    ------------
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as 
        homogeneous coordinates
    '''
    # validate points
    assert points1.shape[0] == points2.shape[0], "no. of points don't match"
    
    # compute centroid of points
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    
    # compute the scaling factor
    s1 = np.sqrt(2 / np.mean(np.sum((points1 - c1) ** 2, axis=1)))
    s2 = np.sqrt(2 / np.mean(np.sum((points2 - c2) ** 2, axis=1)))
    
    # compute the normalization matrix for both the points
    T1 = np.array([
        [s1, 0, -s1 * c1[0]],
        [0, s1, -s1 * c1[1]],
        [0, 0 ,1]
    ])
    T2 = np.array([
        [s2, 0, -s2 * c2[0]],
        [0, s2, -s2 * c2[1]],
        [0, 0, 1]
    ])
    
    # normalize the points
    points1_n = T1 @ points1.T
    points2_n = T2 @ points2.T
    
    # compute the normalized fundamental matrix
    F_n = compute_fundamental_matrix(points1_n.T, points2_n.T)
    
    # de-normalize the fundamental
    return T2.T * F_n * T1

assert (src_pts.shape == dst_pts.shape)

# compute the normalized fundamental matrix 
F = compute_fundamental_matrix_normalized(src_pts, dst_pts)

# validate the fundamental matrix equation
p1 = src_pts.T[:, 0]
p2 = dst_pts.T[:, 0]

# if result if zero, calculation is correct by epipolar constrain
print('result: ', np.round(p2.T @ F @ p1))

# Find Essential Matrix from Fundamental Matrix
print('K: ', K)

EssentialMatrix = K.transpose() @ F @ K

# peform svd on A and find the minimum value of |Af|
U, D, VT = np.linalg.svd(EssentialMatrix, full_matrices=True)
W = np.array([[0, 1, 0],
             [-1, 0, 0],
              [0, 0, 1]], dtype = "double")

RotationMatrix = U @ W @ VT 
determinantR2 = np.linalg.det(RotationMatrix)
if(determinantR2 > 0):
    print('RotationMatrix:', RotationMatrix)

# translation t will be arbirtary magnitiude, we can only know the direction of t
print('Translation up to scale:', U[:,2])
