import cv2
import numpy as np

def phase_correlation_similarity(img1, img2):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Compute phase correlation
    shift, response = cv2.phaseCorrelate(img1, img2)
    # dx, dy = shift
    peak_value = response  # High = very similar (close to 1.0)
    return peak_value > 0.1