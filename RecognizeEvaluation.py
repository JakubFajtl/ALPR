import os
import re
import cv2
import csv  # Added for writing the error file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

def combine_images_side_by_side(images):
    """
    Helper to stack multiple images horizontally with a separator.
    """
    if not images:
        return None
    if len(images) == 1:
        return images[0]

    # resize all to the maximum height found (to keep aspect ratio decent)
    max_h = max(img.shape[0] for img in images)
    resized_imgs = []

    for img in images:
        h, w = img.shape[:2]
        if h != max_h:
            scale = max_h / h
            new_w = int(w * scale)
            # ensure width is at least 1
            new_w = max(1, new_w)
            img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_LINEAR)
        resized_imgs.append(img)

    # ddd a black vertical bar between images
    separator = np.zeros((max_h, 10, 3), dtype=np.uint8)

    # stack
    combined = resized_imgs[0]
    for i in range(1, len(resized_imgs)):
        combined = np.hstack((combined, separator, resized_imgs[i]))

    return combined


def recognition_score(predictions, ground_truth):
    results = []

    # look for predictions within this many frames of the GT
    frame_tolerance = 80

    pred_frames = sorted(predictions.keys())

    for gt_frame, gt_list in ground_truth.items():

        # find "nearby" predictions
        nearby_preds = []
        for p_frame in pred_frames:
            if abs(p_frame - gt_frame) <= frame_tolerance:
                nearby_preds.extend(predictions[p_frame])

        best_score = 0.0
        best_match_text = "None"

        # default placeholder
        plate_crop = np.zeros((100, 300, 3), dtype=np.uint8)
        cv2.putText(plate_crop, "No Image", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for gt_text in gt_list:
            best_score = 0.0
            best_match_text = "None"
            if nearby_preds:
                for pred in nearby_preds:
                    text = pred['text']
                    score = SequenceMatcher(None, gt_text, text).ratio()


                    if score > best_score:
                        best_score = score
                        best_match_text = text
                        # Use the image from this prediction
                        if pred['plate_image'] is not None:
                            plate_crop = pred['plate_image']

            results.append({
                'frame': gt_frame,
                'ground truth': gt_text,
                'best prediction': best_match_text,
                'score': round(best_score, 4),
                'plate_image': plate_crop
            })

    if not results:
        print("No results to evaluate.")
        return []

    total_items = len(results)
    perfect_matches = len([r for r in results if r['score'] == 1.0])
    avg_score = np.mean([r['score'] for r in results])

    print("\n" + "=" * 30)
    print(" EVALUATION RESULTS ")
    print("=" * 30)
    print(f"Total Frames Evaluated : {total_items}")
    print(f"Perfect Matches (1.0)  : {perfect_matches}")
    print(f"Average Accuracy       : {avg_score:.2%}")
    print("-" * 30)

    return results


def save_errors_to_csv(results, output_path):
    """
    Writes a CSV file containing only the frames where prediction failed (score < 1.0).
    """
    # filter for errors
    errors = [r for r in results if r['score'] < 1.0]

    if not errors:
        print("No errors found. CSV will not be created.")
        return

    # ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare data for CSV
    csv_rows = []
    for item in results:
        passS = not (item['score']<1)
        csv_rows.append({
            'Frame': item['frame'],
            'Ground Truth': item['ground truth'],
            'Prediction': item['best prediction'],
            'Score': item['score'],
            'Pass': passS
        })

    # write to CSV
    if csv_rows:
        keys = csv_rows[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved error report to: '{output_path}'")


def save_plots_to_folder(results, output_folder="prediction_plots"):
    error_count = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, item in enumerate(results):
        if item['score'] < 1.0:
            error_count += 1
            plt.figure(figsize=(8, 4))  # slightly wider for multiple plates

            img = item['plate_image']
            if isinstance(img, np.ndarray) and img.ndim == 3:
                img = img[:, :, ::-1]  # BGR to RGB

            plt.imshow(img)

            pred_text = item.get('best prediction', 'None')
            frame_num = item.get('frame', '0')
            raw_score = item.get('score', 0)
            gt = item.get('ground truth', '?')

            title_str = f"Frame: {frame_num}\nGT: {gt} | Pred: {pred_text}\nScore: {raw_score:.2f}"
            plt.title(title_str, fontsize=12, fontweight='bold', color='red')
            plt.axis('off')

            safe_text = re.sub(r'[^a-zA-Z0-9]', '', str(pred_text))
            filename = f"err_F{frame_num}_GT-{gt}_Pred-{safe_text}.png"
            save_path = os.path.join(output_folder, filename)

            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    print(f"\nSaved {error_count} error plots to directory: '{output_folder}'")


def recognition_evaluation(
        ground_truth_csv_path="ground_truth_plate_text.csv",
        prediction_csv_path="dataset/Output.csv",
        cropped_images_dir="training_set/training_output"
):
    # load gt
    if not os.path.exists(ground_truth_csv_path):
        print(f"Error: Ground Truth CSV not found at {ground_truth_csv_path}")
        return

    df_gt = pd.read_csv(ground_truth_csv_path)
    df_gt.columns = df_gt.columns.str.strip()

    if 'Frame Number' in df_gt.columns and 'Text' in df_gt.columns:
        frame_col_gt = 'Frame Number'
        text_col_gt = 'Text'
    else:
        frame_col_gt = next((c for c in df_gt.columns if 'frame' in c.lower()), None)
        text_col_gt = next((c for c in df_gt.columns if 'text' in c.lower() or 'plate' in c.lower()), None)

    ground_truth_dict = {}
    for index, row in df_gt.iterrows():
        try:
            frame_num = int(row[frame_col_gt])
            plate_truth = str(row[text_col_gt]).strip()
            if frame_num not in ground_truth_dict:
                ground_truth_dict[frame_num] = []
            ground_truth_dict[frame_num].append(plate_truth)
        except ValueError:
            continue

    # load predictions and cropped images
    if not os.path.exists(prediction_csv_path):
        print(f"Error: Prediction CSV not found at {prediction_csv_path}")
        return

    print(f"Loading predictions from: {prediction_csv_path}")
    df_pred = pd.read_csv(prediction_csv_path)
    df_pred.columns = df_pred.columns.str.strip()

    if 'Frame no.' in df_pred.columns and 'License plate' in df_pred.columns:
        pred_frame_col = 'Frame no.'
        pred_text_col = 'License plate'
    else:
        pred_frame_col = next((c for c in df_pred.columns if 'Frame' in c), None)
        pred_text_col = next((c for c in df_pred.columns if 'License' in c or 'Plate' in c), None)

    predictions = {}

    search_dirs = [cropped_images_dir, "custom output", "training_set", "images", "."]

    for index, row in df_pred.iterrows():
        try:
            frame_num = int(row[pred_frame_col])
            text_pred = str(row[pred_text_col]).strip()

            found_images = []
            possible_suffixes = [0, 1]

            for suffix in possible_suffixes:
                base_name = f"frame_{frame_num:06d}_{suffix}.jpg"
                for d in search_dirs:
                    full_path = os.path.join(d, base_name)
                    if os.path.exists(full_path):
                        img = cv2.imread(full_path)
                        if img is not None:
                            found_images.append(img)
                            break

            final_img = None
            if found_images:
                final_img = combine_images_side_by_side(found_images)
            else:
                final_img = np.zeros((100, 300, 3), dtype=np.uint8)
                cv2.putText(final_img, f"Crop {frame_num} Missing", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

            if frame_num not in predictions:
                predictions[frame_num] = []

            predictions[frame_num].append({
                "text": text_pred,
                "plate_image": final_img
            })

        except ValueError:
            continue

    print(f"Loaded predictions for {len(predictions)} unique frames.")

    # score save plot, and write error file
    scores = recognition_score(predictions, ground_truth_dict)

    output_dir = "prediction_plots/validation_data"
    save_plots_to_folder(scores, output_dir)

    csv_path = os.path.join(output_dir, "errors.csv")
    save_errors_to_csv(scores, csv_path)


def recognition_evaluation_formated_all(
        ground_truth_csv_path="dataset/groundTruth.csv",
        prediction_csv_path="dataset/Output.csv",
        cropped_images_dir="training_set/training_output",
        exclusion_csv_path="ground_truth_plate_text.csv"
):
    excluded_plates = set()
    if os.path.exists(exclusion_csv_path):
        df_ex = pd.read_csv(exclusion_csv_path)
        ex_text_col = next((c for c in df_ex.columns if 'text' in c.lower() or 'plate' in c.lower()), None)
        if ex_text_col:
            excluded_plates = set(df_ex[ex_text_col].astype(str).str.strip().tolist())
    # load gt
    if not os.path.exists(ground_truth_csv_path):
        print(f"Error: Ground Truth CSV not found at {ground_truth_csv_path}")
        return

    print(f"Loading Ground Truth from: {ground_truth_csv_path}")
    df_gt = pd.read_csv(ground_truth_csv_path)
    df_gt.columns = df_gt.columns.str.strip()

    ground_truth_dict = {}

    if 'First frame' in df_gt.columns and 'Last frame' in df_gt.columns and 'text' in df_gt.columns:
        for index, row in df_gt.iterrows():
            try:
                start_frame = int(row['First frame'])
                end_frame = int(row['Last frame'])
                plate_truth = str(row['text']).strip()
                if plate_truth in excluded_plates: continue  #run only on training frames

                for frame_num in range(start_frame, end_frame + 1):
                    if frame_num not in ground_truth_dict:
                        ground_truth_dict[frame_num] = []
                    ground_truth_dict[frame_num].append(plate_truth)
            except ValueError:
                continue
    else:
        if 'Frame Number' in df_gt.columns and 'Text' in df_gt.columns:
            frame_col_gt = 'Frame Number'
            text_col_gt = 'Text'
        else:
            frame_col_gt = next((c for c in df_gt.columns if 'frame' in c.lower()), None)
            text_col_gt = next((c for c in df_gt.columns if 'text' in c.lower() or 'plate' in c.lower()), None)

        if frame_col_gt and text_col_gt:
            for index, row in df_gt.iterrows():
                try:
                    frame_num = int(row[frame_col_gt])
                    plate_truth = str(row[text_col_gt]).strip()
                    if plate_truth in excluded_plates: continue #run only on training frames
                    if frame_num not in ground_truth_dict:
                        ground_truth_dict[frame_num] = []
                    ground_truth_dict[frame_num].append(plate_truth)
                except ValueError:
                    continue
        else:
            print("Error: Could not determine Ground Truth columns.")
            return

    print(f"Loaded Ground Truth for {len(ground_truth_dict)} frames.")

    # predicitons and croipped images
    if not os.path.exists(prediction_csv_path):
        print(f"Error: Prediction CSV not found at {prediction_csv_path}")
        return

    print(f"Loading predictions from: {prediction_csv_path}")
    df_pred = pd.read_csv(prediction_csv_path)
    df_pred.columns = df_pred.columns.str.strip()

    if 'Frame no.' in df_pred.columns and 'License plate' in df_pred.columns:
        pred_frame_col = 'Frame no.'
        pred_text_col = 'License plate'
    else:
        pred_frame_col = next((c for c in df_pred.columns if 'Frame' in c), None)
        pred_text_col = next((c for c in df_pred.columns if 'License' in c or 'Plate' in c), None)

    if not pred_frame_col or not pred_text_col:
        print("Error: Could not determine Prediction columns.")
        return

    predictions = {}
    search_dirs = [cropped_images_dir, "custom output", "training_set", "images", "."]

    for index, row in df_pred.iterrows():
        try:
            frame_num = int(row[pred_frame_col])
            text_pred = str(row[pred_text_col]).strip()

            found_images = []
            possible_suffixes = [0, 1]

            for suffix in possible_suffixes:
                base_name = f"frame_{frame_num:06d}_{suffix}.jpg"
                for d in search_dirs:
                    full_path = os.path.join(d, base_name)
                    if os.path.exists(full_path):
                        img = cv2.imread(full_path)
                        if img is not None:
                            found_images.append(img)
                            break

            final_img = None
            if found_images:
                final_img = combine_images_side_by_side(found_images)
            else:
                final_img = np.zeros((100, 300, 3), dtype=np.uint8)
                cv2.putText(final_img, f"Crop {frame_num} Missing", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

            if frame_num not in predictions:
                predictions[frame_num] = []

            predictions[frame_num].append({
                "text": text_pred,
                "plate_image": final_img
            })

        except ValueError:
            continue

    print(f"Loaded predictions for {len(predictions)} unique frames.")

    # score & save Plots & error file
    scores = recognition_score(predictions, ground_truth_dict)

    output_dir = "prediction_plots/all_data"
    save_plots_to_folder(scores, output_dir)

    csv_path = os.path.join(output_dir, "errors.csv")
    save_errors_to_csv(scores, csv_path)


if __name__ == "__main__":
    cropped_images_dir = "training_set/training_output"
    prediction_csv_path = "Output.csv"
    validation_csv_path = "ground_truth_plate_text.csv"
    recognition_evaluation(
        ground_truth_csv_path=validation_csv_path,
        prediction_csv_path=prediction_csv_path,
        cropped_images_dir="training_set/training_output"
    )
    recognition_evaluation_formated_all(
        ground_truth_csv_path="dataset/groundTruth.csv",
        prediction_csv_path=prediction_csv_path,
        cropped_images_dir=cropped_images_dir,
        exclusion_csv_path = validation_csv_path
    )