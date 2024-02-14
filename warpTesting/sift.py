import cv2
import numpy as np

def find_warping_points_sift(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Use BFMatcher with default parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Get corresponding points in both images
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts

# Example usage:
image1= cv2.imread("warpTesting/aryansh6.jpg")
image2=cv2.resize(image1,(741,800))

src_pts, dst_pts = find_warping_points_sift(image2, image1)

M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply the perspective transformation to the image
result = cv2.warpPerspective(image1, M, (image1.shape[1], image1.shape[0]))

cv2.imshow("warpTesting/siftwarp.jpg",result)
cv2.waitKey(2000)
