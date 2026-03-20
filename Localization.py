import cv2
import numpy as np

def plate_detection(image):
    """
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
    """
    # TODO: Replace the below lines with your code.
    # get plate coordinates
    top, bottom, left, right, double, center, rotation = plate_coordinates(image)
    # crop image to only return plate
    cropped = crop(image, top, bottom, left, right)
    result = []
    # if the crop is too long we probably ran into 2 plates in one frame
    if cropped.shape[0] == 480:
        return result
    if double:
        mid_point =(image.shape[1])//2
        # we split the image into two halves
        first_half = image[:, :mid_point]
        second_half = image[:, mid_point:]
        # we figure out the plate location on each half
        t1, b1, l1, r1, d1, c1, ro1 = plate_coordinates(first_half)
        first_rotated = rotate_image(first_half, c1, ro1)
        t1, b1, l1, r1, d1, c1, ro1 = plate_coordinates(first_rotated)
        t2, b2, l2, r2, d2, c2, ro2 = plate_coordinates(second_half)
        second_rotated = rotate_image(second_half, c2, ro2)
        t2, b2, l2, r2, d2, c2, ro2 = plate_coordinates(second_rotated)
        
        # we crop each half
        first_plate = crop(first_rotated,t1, b1, l1, r1)
        second_plate = crop(second_rotated, t2, b2, l2, r2)

        # we calculate the global coordinates for each plate
        p1_global = [top + t1, top + b1, left + l1, left + r1, c1, ro1]
        p2_global = [top + t2, top + b2, left + mid_point + l2, left + mid_point + r2, c2, ro2]
        candidates = [(first_plate, p1_global), (second_plate, p2_global)]
    else:
        # else we just add the plate
        rotated = rotate_image(image, center, rotation)
        top, bottom, left, right, double, center, rotation = plate_coordinates(rotated)
        candidates = [(crop(rotated, top, bottom, left, right), (top, bottom, left, right, center, rotation))]

    for plate_img, coords in candidates:
        # we discard all plates that have a higher aspect ratio than 0.5 or those which span the full image
        if plate_img.shape[0]/plate_img.shape[1] < 0.5 and plate_img.shape[0] != 480:
            result.append({'img': rotate_image(plate_img, coords[4], coords[5]), 'coords': coords})

    return result

def evaluation_ofLocalization(frames):
    predictions = []
    #save prediction of localization to compare with ground truth
    for fr in frames:
        current_frame_image = fr[0]
        frame_num = fr[1]

        #get localized frames
        plate_results = plate_detection(current_frame_image)

        for result in plate_results:
            coords = result['coords']
            predictions.append([frame_num, coords[0], coords[1], coords[2], coords[3]])

    return predictions


def plate_coordinates(image):

    processed = np.array(image)

    # preprocessing, currently empty
    processed = np.array(preprocess(processed))
    # color mask
    processed = np.array(apply_color_mask(processed))
    # morphology mask
    processed = np.array(apply_opening_mask(processed))
    (top, bottom, left, right, double, center, rotation) = image_coordinates(processed)

    return top, bottom, left, right, double, center, rotation

def apply_color_mask(image):
    # TODO: Replace the below lines with your code.
    # Define color range

    conv_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colorMin = np.array([18,115,100])
    colorMax = np.array([50,255,255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)


    mask = cv2.inRange(conv_HSV, colorMin, colorMax)

    masked = cv2.bitwise_and(image, image, mask=mask)

    return masked

def apply_opening_mask(image):
    # creating mask, this can probably be done in a simpler way lol
    conv_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colorMin = np.array([1,1,1])
    colorMax = np.array([179,255,255])
    mask = cv2.inRange(conv_HSV, colorMin, colorMax)

    structuring_element = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=np.uint8)
    structuring_element_2 = np.array([[1,1,1],[1,1,1], [1,1,1]], dtype=np.uint8)

    # experimental operations
    # structuring_element = np.array([[1,1], [1,1]], dtype=np.uint8)
    # mask = cv2.erode(mask, structuring_element, iterations = 1)
    # mask = cv2.dilate(mask, structuring_element, iterations = 3)
    # mask = cv2.erode(mask, structuring_element, iterations = 2)
    # mask = cv2.erode(mask, structuring_element_2, iterations = 10)
    # mask = cv2.dilate(mask, structuring_element_2, iterations = 10)
    # mask = cv2.erode(mask, structuring_element, iterations = 2)
    # mask = cv2.dilate(mask, structuring_element, iterations = 2)

    mask = cv2.dilate(mask, structuring_element_2, iterations = 10)
    mask = cv2.erode(mask, structuring_element_2, iterations = 10)

    mask = cv2.erode(mask, structuring_element, iterations = 3)
    mask = cv2.dilate(mask, structuring_element, iterations = 3)

    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

def image_coordinates(image):
    top = 0
    bottom = len(image)
    left = 0
    right = len(image[0])

    # top
    for i in range(image.shape[0]):
        if np.any(image[i, :, 0] != 0):
            # print("found", i)
            top = i
            break

    # bottom
    for i in range(image.shape[0] - 1, -1, -1):
        if np.any(image[i, :, 0] != 0):
            bottom = i
            break

    # left
    for j in range(image.shape[1]):
        if np.any(image[:, j, 0] != 0):
            left = j
            break

    # right
    for j in range(image.shape[1] - 1, -1, -1):
        if np.any(image[:, j, 0] != 0):
            right = j
            break

    cropped_image = image[top:bottom, left:right]
    double = True
    # if we find any non-black pixel on a vertical line in the middle of the image then it is only one plate
    if np.any(cropped_image[:, cropped_image.shape[1]//2] != 0):
        double = False
    # we return boundaries for further manipulation
    center, rotation = get_center_and_rotation((cropped_image[:, :, 0] != 0).astype(int) )
    return (top, bottom, left, right, double, center, rotation)


def preprocess(image):
    return second(first(image))

def second(image):
    # white balance bh scaling down brightness
    percentile = 99
    img_float = image.astype(np.float32)
    intensity = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = np.percentile(intensity, percentile)
    mask = intensity >= threshold

    avg_b = np.mean(img_float[:, :, 0][mask])
    avg_g = np.mean(img_float[:, :, 1][mask])
    avg_r = np.mean(img_float[:, :, 2][mask])

    max_avg = max(avg_b, avg_g, avg_r)
    gain_b = max_avg / (avg_b + 1e-6)
    gain_g = max_avg / (avg_g + 1e-6)
    gain_r = max_avg / (avg_r + 1e-6)
    
    balanced = img_float.copy()
    balanced[:, :, 0] *= gain_b
    balanced[:, :, 1] *= gain_g
    balanced[:, :, 2] *= gain_r
    
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced

def first(image):
    # white balance using grey world assumption
    b, g, r = cv2.split(image.astype(np.float32))
    
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    
    epsilon = 1e-6
    scale_b = (mean_g + mean_r) / (2 * (mean_b + epsilon))
    scale_g = (mean_b + mean_r) / (2 * (mean_g + epsilon))
    scale_r = (mean_b + mean_g) / (2 * (mean_r + epsilon))
    
    b = np.clip(b * scale_b, 0, 255)
    g = np.clip(g * scale_g, 0, 255)
    r = np.clip(r * scale_r, 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)

def crop(image, top, bottom, left, right):
    return image[top:bottom, left:right]

def get_center_and_rotation(binary_image):
    # we have to convert to an array of points so that when we center the image it can go into negative values
    ys, xs = np.where(binary_image == 1)
    points = np.column_stack((xs, ys))

    # center the points
    center = points.mean(axis=0, dtype=np.int32)
    center = center[0], center[1]
    centered = points - center

    # get covariance matrix
    cov = np.cov(centered, rowvar=False)

    # get the eigenvectors 
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
 
    # get the bigger eigenvector which is the one we will be referencing to get our angle
    axis = eigenvectors[:, np.argmax(eigenvalues)]

    # calculate plate rotation angle from the axis
    rotation = np.arctan2(axis[1], axis[0])

    return center, rotation

def rotate_image(image, center, angle):
    angle_deg = np.degrees(angle)

    # get transformation matrix
    rotation_matrix = cv2.getRotationMatrix2D(np.array(center, dtype=np.float32), angle_deg - 180, scale=1.0)

    # apply transform
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        borderValue=0
    )

    return rotated