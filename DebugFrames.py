import numpy as np
import pandas as pd
import cv2
import os

from RecognizeEvaluation import combine_images_side_by_side


def get_frames_from_error_csv(csv_path):
    """
    Reads 'errors.csv' and returns a dictionary:
    {
      frame_number: {
          'prediction': 'ABC-123',
          'ground_truth': 'ABC-123',
          'score': 0.85
      }
    }
    """
    print(f"Loading error frames from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        # clean column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # check if required columns exist
        if 'Frame' not in df.columns or 'Prediction' not in df.columns:
            print("Error: CSV must contain 'Frame' and 'Prediction' columns.")
            print(f"Found columns: {df.columns.tolist()}")
            return {}

        # build the dictionary
        error_data = {}
        for index, row in df.iterrows():
            frame_num = int(row['Frame'])

            error_data[frame_num] = {
                'prediction': str(row['Prediction']),
                'ground_truth': str(row.get('Ground Truth', 'Unknown')),  # detailed info if available
                'score': float(row.get('Score', 0.0))  # detailed info if available
            }

        print(f"Found {len(error_data)} frames marked for debugging.")
        return error_data

    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return {}


def debug_run(image_dir, error_csv_path):
    # get the dictionary of errors
    error_dict = get_frames_from_error_csv(error_csv_path)

    if not error_dict:
        print("No error frames found to process.")
        return

    sorted_frames = sorted(error_dict.keys())
    print(f"Starting Debug Visualization for frames: {sorted_frames}")

    # loop specifically through these frames
    for frame_num in sorted_frames:

        frame_info = error_dict[frame_num]
        pred_text = frame_info['prediction']
        gt_text = frame_info['ground_truth']
        score = frame_info['score']

        # image loading logic
        found_images = []
        possible_suffixes = [0, 1]

        for suffix in possible_suffixes:
            base_name = f"frame_{frame_num:06d}_{suffix}.jpg"
            full_path = os.path.join(image_dir, base_name)

            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                if img is not None:
                    found_images.append(img)


        if found_images:
            final_img = combine_images_side_by_side(found_images)
        else:
            final_img = np.zeros((100, 300, 3), dtype=np.uint8)
            cv2.putText(final_img, "IMG MISSING", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        print(f"\n" + "=" * 40)
        print(f" DEBUGGING FRAME : {frame_num}")
        print(f" PREDICTION      : {pred_text}")
        print(f" GROUND TRUTH    : {gt_text}")
        print(f" SCORE           : {score}")
        print("=" * 40)


if __name__ == "__main__":
    target_csv = "prediction_plots/all_data/errors.csv"

    debug_run(
        image_dir="training_set/training_output",
        error_csv_path=target_csv
    )