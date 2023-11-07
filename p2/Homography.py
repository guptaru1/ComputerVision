import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.transform import ProjectiveTransform, warp

left_image = cv2.imread('bbb_left.jpg', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('bbb_right.jpg', cv2.IMREAD_GRAYSCALE)

left_img = cv2.imread('bbb_left.jpg')
left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

right_img = cv2.imread('bbb_right.jpg')
right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(left_image, cmap='gray')
plt.title('Left Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(right_image, cmap='gray')
plt.title('Right Image')
plt.axis('off')


# Create SIFT or SURF object (choose one)
sift = cv2.SIFT_create()
# surf = cv2.xfeatures2d.SURF_create()

# Detect keypoints and compute descriptors
keypoints_left, descriptors_left = sift.detectAndCompute(left_image, None)
keypoints_right, descriptors_right = sift.detectAndCompute(right_image, None)

# Draw keypoints on the images
left_image_with_keypoints = cv2.drawKeypoints(left_image, keypoints_left, None)
right_image_with_keypoints = cv2.drawKeypoints(right_image, keypoints_right, None)

# Display images with keypoints
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(left_image_with_keypoints, cmap='gray')
plt.title('Left Image with SIFT Keypoints')
plt.axis('off')

plt.subplot(122)
plt.imshow(right_image_with_keypoints, cmap='gray')
plt.title('Right Image with SIFT Keypoints')
plt.axis('off')

plt.show()


left_mean = np.mean(descriptors_left, axis=0)
std_left = np.std(descriptors_left, axis=0)
    
des_left = (descriptors_left - left_mean) / std_left
    
right_mean = np.mean(descriptors_right, axis = 0)
std_right = np.std(descriptors_right, axis = 0)
    
des_right = (descriptors_right - right_mean)/std_right
    
matching_scores = []
    
for i in range(len(descriptors_left)):
        for j in range(len(descriptors_right)):
            
            euclidean_distance = np.linalg.norm(descriptors_left[i] - descriptors_right[j])
            matching_scores.append(euclidean_distance)
    
matching_scores = np.array(matching_scores)

matching_scores_new = np.linalg.norm(descriptors_left[:, np.newaxis, :] - descriptors_right, axis=2)
print("Matching_Scores", matching_scores_new)


def get_pair_min_distances(num_pairs, matching_scores,kp1,kp2):
    dist_image_1 = []
    dist_image_2 = []
    min_distance = []
    
    print("Matching scores shape", matching_scores.shape)
    
    for desc_1 in range(matching_scores.shape[0]):
        dist_image_1.append(min(matching_scores[desc_1,:]))
    for desc_2 in range(matching_scores.shape[1]):
        dist_image_2.append(min(matching_scores[:,desc_2]))
    
    for i in range(len(dist_image_1)):
        for j in range(len(dist_image_2)):
            if dist_image_1[i] == dist_image_2[j]:
                min_distance.append([dist_image_1[i],i,j])
    min_distance = sorted(min_distance)[:num_pairs]
    data = []
    for item in min_distance:
        x1, y1 = kp1[item[1]].pt
        x2, y2 = kp2[item[2]].pt
        data.append([x1, y1, x2, y2])
    data = np.array(data, dtype = 'int')
    return data
            
        
pairwise_distances = get_pair_min_distances(500, matching_scores_new,keypoints_left,keypoints_right)
    
def homography(four_matches):
    #direct linear transform and singular value decomposition
    A = []
    for match in four_matches:
        x_1, y_1 = match[0], match[1] # x, y
        x_2, y_2 = match[2], match[3]
        A.append([x_1, y_1, 1, 0, 0, 0, -x_1*x_2, -y_1*x_2, -x_2])
        A.append([0, 0, 0, x_1, y_1, 1, -x_1*y_2, -y_1*y_2, -y_2])
        
    U, S, V = np.linalg.svd(np.matrix(A))
    h = np.reshape(V[8], (3, 3))
    h = (1/h.item(8)) * h
    return h

def residual(match, H):
    p1 = np.transpose(np.matrix([match[0], match[1], 1]))
    estimatep2 = np.dot(H, p1)
    #normalise all points of estimatedp2 to ensure the homogenous corordiante is set to 1
    estimatep2 = (1/estimatep2.item(2))*estimatep2
    
    p2 = np.transpose(np.matrix([match[2], match[3], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

def RANSAC(data, iteration = 1500,threshold=0.6):
    best_model, best_residual, num_best_inliers = None, None, []
    for i in range(iteration):
        match1 = data[random.randrange(0, len(data))]
        match2 = data[random.randrange(0, len(data))]
        fourMatches = np.vstack((match1, match2))
        match3 = data[random.randrange(0, len(data))]
        fourMatches = np.vstack((fourMatches, match3))
        match4 = data[random.randrange(0, len(data))]
        #four matches as homography has 8 variables so need min 4 points
        fourMatches = np.vstack((fourMatches, match4))
        
        H = homography(fourMatches)
        
        #  avoid dividing by zero 
        #if np.linalg.matrix_rank(H) < 3:
            #print("Coming in here")
            #continue
            
        #good matching points
        inliers = []
        all_residual = 0
        #go throught all the match points using the matrix h obtained
        for i in range(len(data)):
            
            r = residual(data[i], H)
            if r < 6:
                
                inliers.append(data[i])
            all_residual += r

        #check if this was the ebst model found yet
        if len(inliers) > len(num_best_inliers):
            num_best_inliers = inliers
            best_model = H
            best_residual = all_residual / len(data)
            print("best model ", best_model,"Best residual", best_residual, "Max inliers: ", len(num_best_inliers))
        #Threshold condition
        if len(num_best_inliers) > (len(data)*threshold):
            break
            
    return best_model,num_best_inliers, best_residual
        
        #construct a sampel of s from teh data points
        
# Initialize the feature matcher using brute-force matching
#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using brute-force matching
#matches_bf = bf.match(descriptors1, descriptors2)

def warp_images(H, img1, img2):
    """
    stitch images together according to the homography
    """
   
    height = img1.shape[0]
    width = img1.shape[1] + img2.shape[1]
    depth = img1.shape[2]
    trans = ProjectiveTransform(H)
    #apply homography to the image 2
    img2_trans = warp(img2, trans, output_shape=(height,width,depth))
    #convert pixel into the range of 0 to 255
    img2_trans = img2_trans * 255.
    img2_trans = img2_trans.astype('int')
    
    #create a sample image
    img1_trans = np.zeros((height, width, depth), dtype = 'int')
    #make the columns upto a certain point of image1, the left part is all image 1
    
    img1_trans[:,:img1.shape[1]] = img1
    
    #final image to return
    warped_img = np.zeros((height, width, depth), dtype = 'int')
    #rows
    for i in range(warped_img.shape[0]):
        #cols
        for j in range(warped_img.shape[1]):
           # if (img2_trans[i][j] != 0).all() and (img1_trans[i][j] != 0).all():
                #warped_img[i][j] = (img2_trans[i][j] + img1_trans[i][j]) / 2
            if (img2_trans[i][j] != 0).all():
                warped_img[i][j] = img2_trans[i][j]
            else:
                warped_img[i][j] = img1_trans[i][j]
#     plt.figure()
#     plt.imshow(warped_img)
#     plt.show()
    return warped_img

def w_images( H,left,right):

    
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)
    
    height_l, width_l, channel_l = left.shape
    #corners of current left image
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    new_corners = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(new_corners).T 
    
    #represent correct pixel positions in the cartesian co-ordianate for pixels
    cartesian_x = corners_new[0]/corners_new[2]
    cartesian_y = corners_new[1] / corners_new[2]
    
    y_min = min(cartesian_y)
    x_min = min(cartesian_x)
    
    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)
    
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
    
    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
    
        # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image




# Check if this script is the main script being run
if __name__ == "__main__":
    print("Hello world")
    #call the function
    data = get_pair_min_distances(250, matching_scores_new,keypoints_left,keypoints_right)
    H, max_inliers, best_model_errors = RANSAC(data)
    matches = max_inliers
    # Convert keypoint objects to numpy arrays
    #keypoints1 = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    #keypoints2 = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Create an output image that combines the two input images side by side
    #img_matches = np.hstack((img1, img2))

    # Draw the matches
    #img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
    # Convert the BGR image to RGB
    #img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    # Display the matches using plt.imshow
    #print("Best model:", best_model)
    #print("Average residual:", np.average(best_model_errors))
    #print("Inliers:", len(max_inliers))
    
    #matched_image = cv2.drawMatches(left_image, keypoints_left, right_image, keypoints_right, max_inliers, None)
    #matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    #plt.imshow(warp_images(H, left_rgb, right_rgb,))
    plt.imshow(warp_images(H, left_rgb, right_rgb,))
    
    #plt.imshow(matched_image_rgb)  # Display the image
    plt.axis('off')  # Turn off axis labels and ticks (optional)
    plt.show()  # Show the image
