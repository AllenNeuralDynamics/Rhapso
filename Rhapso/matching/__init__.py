def main(args):
    print("Hello, Matching!")
    print("Received the following arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")



# import os
# import cv2 as cv
# import numpy as np
# import argparse

# def main(args):
#     print("Hello, World!")
#     print("Received the following arguments:")
#     for arg in vars(args):
#         print(f"{arg}: {getattr(args, arg)}")

#     # Process TIFF images and perform feature matching with homography
#     featureMatchingHomeography_tiff(args.tiffPath)

# def featureMatchingHomeography_tiff(tiff_path):
#     """Feature matching with homography using images from specified TIFF files and z-indexes."""
    
#     # Define TIFF file path for both images (assumes both slices come from the same file)
#     img1_path = tiff_path
#     img2_path = tiff_path

#     # Specify z-index slices for the two images being compared
#     z_index1 = 39
#     z_index2 = 40

#     # Read images from the TIFF file using OpenCV
#     img1 = cv.imreadmulti(img1_path, flags=cv.IMREAD_UNCHANGED)[1][z_index1]
#     img2 = cv.imreadmulti(img2_path, flags=cv.IMREAD_UNCHANGED)[1][z_index2]

#     if img1 is None or img2 is None:
#         print("Error: Could not read one or both of the images.")
#         return

#     # Create a SIFT detector for detecting keypoints and computing descriptors
#     sift = cv.SIFT_create()
#     keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

#     # Set up the FLANN-based matcher
#     # FLANN_INDEX_KDTREE is an algorithm optimized for high-dimensional data searches
#     FLANN_INDEX_KDTREE = 1
#     indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # KD-Tree with 5 trees
#     searchParams = dict(checks=50)  # Limit the number of comparisons for efficiency
#     flann = cv.FlannBasedMatcher(indexParams, searchParams)
    
#     # Perform K-Nearest Neighbors (KNN) matching (finding the 2 best matches per descriptor)
#     nNeighbors = 2
#     matches = flann.knnMatch(descriptors1, descriptors2, k=nNeighbors)

#     # Apply Lowe's ratio test to filter good matches
#     goodMatches = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:  # Only keep matches where the best match is significantly better than the second-best
#             goodMatches.append(m)

#     print(f"Number of matches before RANSAC: {len(goodMatches)}")

#     # Apply RANSAC to filter out outliers and compute homography transformation
#     minGoodMatches = 20

#     if len(goodMatches) > minGoodMatches:
#         # Extract matched keypoints from both images
#         src_pts = np.float32([keypoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        
#         # Compute homography matrix using RANSAC to reject outliers
#         errorThreshold = 1  # Maximum pixel error allowed for inliers
#         M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, errorThreshold)
#         matchesMask = mask.ravel().tolist()
        
#         # Warp a rectangle representing img1's borders onto img2 to visualize alignment
#         h, w = img1.shape
#         imgBorder = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
#         warpedImgBorder = cv.perspectiveTransform(imgBorder, M)
#         img2 = cv.polylines(img2, [np.int32(warpedImgBorder)], True, 255, 3, cv.LINE_AA)
        
#         print(f"Number of matches after RANSAC: {np.sum(matchesMask)}")
#     else:
#         print("Not enough matches")
#         matchesMask = None

#     # Define parameters for drawing matches
#     green = (0, 255, 0)  # Color for good matches
#     drawParams = dict(
#         matchColor=green,  # Matches are drawn in green
#         singlePointColor=None,  # No single keypoints displayed
#         matchesMask=matchesMask,  # Apply RANSAC mask to show only inliers
#         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # Exclude unmatched keypoints
#     )

#     # Draw matches between the two images
#     imgMatch = cv.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, None, **drawParams)

#     # Create a resizable window to display the matched features
#     cv.namedWindow('Image Matches', cv.WINDOW_NORMAL)
#     cv.imshow('Image Matches', imgMatch)

#     # Enable zooming functionality using mouse scroll
#     def zoom(event, x, y, flags, param):
#         if event == cv.EVENT_MOUSEWHEEL:
#             if flags > 0:
#                 cv.resizeWindow('Image Matches', int(imgMatch.shape[1] * 1.1), int(imgMatch.shape[0] * 1.1))
#             else:
#                 cv.resizeWindow('Image Matches', int(imgMatch.shape[1] * 0.9), int(imgMatch.shape[0] * 0.9))

#     cv.setMouseCallback('Image Matches', zoom)

#     # Wait for user interaction before closing the window
#     key = cv.waitKey(0)
#     if key == ord('q'):
#         cv.destroyAllWindows()  # Close the window when 'q' is pressed
