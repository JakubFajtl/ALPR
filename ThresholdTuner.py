import cv2
import os
import numpy as np

# change this to point to a frame for debug
TEST_IMAGE_PATH = "training_set/training_output/frame_001672_0.jpg"


def nothing(x):
    pass


def tune_threshold():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Could not find image at {TEST_IMAGE_PATH}")
        return

    # load and prep image (grayscale & aspect ratio resize)
    original = cv2.imread(TEST_IMAGE_PATH)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # standardize height to 80px (same as your pipeline)
    target_height = 80
    aspect_ratio = gray.shape[1] / float(gray.shape[0])
    new_width = int(target_height * aspect_ratio)
    img = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_LINEAR)

    cv2.namedWindow('Threshold Tuner')

    # create Trackbars
    # Mode: 0 = Otsu (Auto), 1 = Adaptive Mean, 2 = Adaptive Gaussian, 3 = Manual Binary
    cv2.createTrackbar('Mode', 'Threshold Tuner', 0, 3, nothing)

    # note - block Size (must be odd, min 3)
    cv2.createTrackbar('Block Size', 'Threshold Tuner', 11, 50, nothing)

    # C Constant (for Adaptive) or Threshold (for Manual)
    cv2.createTrackbar('C / Thresh', 'Threshold Tuner', 9, 255, nothing)

    while True:
        # get current positions
        mode = cv2.getTrackbarPos('Mode', 'Threshold Tuner')
        block = cv2.getTrackbarPos('Block Size', 'Threshold Tuner')
        c_val = cv2.getTrackbarPos('C / Thresh', 'Threshold Tuner')

        if block % 2 == 0: block += 1
        if block < 3: block = 3

        binary = None
        status_text = ""

        if mode == 0:
            status_text = "Mode: Otsu (Auto)"
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        elif mode == 1:
            status_text = f"Mode: Adaptive Mean | Block: {block} | C: {c_val}"
            binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, block, c_val)

        elif mode == 2:
            status_text = f"Mode: Adaptive Gauss | Block: {block} | C: {c_val}"
            binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, block, c_val)

        elif mode == 3:
            status_text = f"Mode: Manual Binary | Thresh: {c_val}"
            _, binary = cv2.threshold(img, c_val, 255, cv2.THRESH_BINARY_INV)

        # visualization
        # convert binary to BGR so we can write colored text
        display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # show original (small) in corner for comparison
        orig_small = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = display.shape[:2]
        display[0:h, 0:w] = display  # filling placeholder

        # add Text
        cv2.putText(display, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display, "Press 'q' to quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # stack images side-by-side
        final_view = np.hstack((orig_small, display))

        cv2.imshow('Threshold Tuner', final_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    tune_threshold()