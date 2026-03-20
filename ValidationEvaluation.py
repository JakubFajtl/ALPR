import cv2
import os
import numpy as np
import Localization
import pandas as pd

def evaluate_validation_set(frames,
                            ground_truth_csv_path="ground_truth_localization.csv"):
    #load ground truth
    df = pd.read_csv(ground_truth_csv_path)

    #create dictionary for ground truth
    ground_truth_dict = {}
    for index, row in df.iterrows():
        frame_num = int(row["Frame Number"])
        box = (row["Top"], row["Bottom"], row["Left"], row["Right"])
        if box == (0, 0, 0, 0): continue #frame is invalid so we dont need to check it
        if frame_num not in ground_truth_dict:
            ground_truth_dict[frame_num] = []
        ground_truth_dict[frame_num].append(box)

    #run localization on validation frames, output is a list of predicitons for box
    # as [frame_num, top, bottom, left, right],
    localization_predictions = Localization.evaluation_ofLocalization(frames)

    #create dictionary for predictions
    prediction_dict = {}
    for prediction in localization_predictions:
        frame_num = prediction[0]
        box = (prediction[1:])
        if frame_num not in prediction_dict:
            prediction_dict[frame_num] = []
        prediction_dict[frame_num].append(box)

    #frame numbers of frames from other categories
    categoryIV = [1824,1896,2016]
    categoryIII = [1584,1608,1680]
    #calculate IoU for each prediction
    iou_scores_reg = []
    iou_scores_catIII = []
    iou_scores_catIV = []

    #we loop through each frame that has ground truth
    for frame_num, gt_boxes in ground_truth_dict.items():
        #get prediction boxes for frame
        pred_boxes = prediction_dict.get(frame_num, [])
        #for each ground truth box we find the prediction in image that fits it most and use that plate for calculations
        for gt_box in gt_boxes:
            best_iou = 0
            best_pred_box = "None"
            if pred_boxes:
                #find most compatible guess, aka the prediciton for this gt plate
                for pred_box in pred_boxes:
                    iou = calculateIou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_box = pred_box  # Remember which box gave this score

            if frame_num in categoryIII:
                iou_scores_catIII.append(best_iou)
            elif frame_num in categoryIV:
                iou_scores_catIV.append(best_iou)
            else:
                iou_scores_reg.append(best_iou)

            clean_gt = [int(x) for x in gt_box]
            clean_pred = [int(x) for x in best_pred_box] if best_pred_box != "None" else "None"
            print(f"Frame {frame_num}: GT {clean_gt} matches Pred {clean_pred} with IoU: {best_iou:.4f}")

    #combine results for average IoU
    iou_scores = iou_scores_reg + iou_scores_catIII + iou_scores_catIV
    if len(iou_scores) == 0:
        print("No valid predictions found")
        return
    average_iou = np.mean(iou_scores)
    print(f"Number of valid frame predictions: {len(iou_scores)}")
    print(f"Number of regular plates: {len(iou_scores_reg)}")
    print(f"Number of double plates: {len(iou_scores_catIII)}")
    print(f"Number of white plates: {len(iou_scores_catIV)}")
    print(f"Average IoU: {np.mean(iou_scores)}")
    print(f"Average IoU of regular plates: {np.mean(iou_scores_reg)}")
    print(f"Average IoU of double plates: {np.mean(iou_scores_catIII)}")
    print(f"Average IoU of white plates: {np.mean(iou_scores_catIV)}")


def validationSetup(frames):
    # save validation frames
    validation_dir = "validationSetStuff/validation_frames"
    os.makedirs(validation_dir, exist_ok=True)

    # save log
    validation_log_path = "validationSetStuff/validation_frames.csv"
    log_file = open(validation_log_path, "w")
    log_file.write("Frame No.,Timestamp(seconds),Image Path\n")

    # saving validation images for ground truth
    for v_frame, v_num, v_ts in frames:
        filename = os.path.join(validation_dir, f"frame_{v_num:06d}.jpg")
        cv2.imwrite(filename, v_frame)
        log_file.write(f"{v_num},{v_ts},{filename}\n")

    #evaluate localization with Ground Truths
    evaluate_validation_set(frames)
    validationOutput(frames)

#function to run localization on validation frames, for use in manual checks !not for evaluation!
def validationOutput(frames):
    output_dir = "validationSetStuff/validation_output"
    os.makedirs(output_dir, exist_ok=True)
    for fr in frames:
        current_frame_image = np.array(fr[0])
        current_frame_num = fr[1]
        #localize the plate and return cropped localized plate
        processed = Localization.plate_detection(current_frame_image)
        for i, plate in enumerate(processed):
            # write into validation_output folder to see results of the validation for checks
            filename = os.path.join(output_dir, f"frame_{current_frame_num:06d}_{i}.jpg")
            cv2.imwrite(filename, plate['img'])  # since result is dictionary with image and global coords

#function to calculate IoU between two boxes
def calculateIou(boxA, boxB):
    #accepts boxes as (top, bottom, left, right)
    y1 = max(boxA[0], boxB[0])
    y2 = min(boxA[1], boxB[1])
    x1 = max(boxA[2], boxB[2])
    x2 = min(boxA[3], boxB[3])

    #intersetction area, max(0, ) is to prevent negative values if not overlapping
    inter_area = max(0,x2 - x1) * max(0,y2 - y1)

    #union area
    boxA_area = (boxA[1] - boxA[0]) * (boxA[3] - boxA[2])
    boxB_area = (boxB[1] - boxB[0]) * (boxB[3] - boxB[2])
    union_area = boxA_area + boxB_area - inter_area

    #IoU
    iou = inter_area / union_area
    return iou
