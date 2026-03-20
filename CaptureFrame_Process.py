import cv2
import os
import pandas as pd
import numpy as np

import Localization
import ValidationEvaluation

from difflib import SequenceMatcher
from collections import Counter
import Recognize


def get_best_predictions_clustering(prediction_list,fps):
    if not prediction_list:
        return []

    # filter Noise
    valid_preds = [p for p in prediction_list if len(p[0]) >= 8 and "?" not in p[0]]
    if not valid_preds:
        return []

    # group similar strings
    clusters = []
    match_threshold = 0.8

    for pred in valid_preds:
        text = pred[0]
        found_cluster = False
        for clust in clusters:
            # Check similarity against the cluster's representative
            if SequenceMatcher(None, text, clust['rep']).ratio() > match_threshold:
                clust['votes'].append(pred)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append({'votes': [pred], 'rep': text})

    # 3. Process Clusters & Fix Timing
    initial_results = []
    internal_frame_gap_threshold = int(fps*2)  # Max gap allowed within one "event"
    min_votes = 6  # Minimum detections to count as valid

    for clust in clusters:
        # sort votes by frame number
        votes = sorted(clust['votes'], key=lambda x: x[1])

        if not votes:
            continue

        # split logic (Separate distinct events in one cluster)
        events = []
        current_event = [votes[0]]

        for k in range(1, len(votes)):
            prev_frame = votes[k - 1][1]
            curr_frame = votes[k][1]

            if curr_frame - prev_frame > internal_frame_gap_threshold:
                events.append(current_event)
                current_event = []

            current_event.append(votes[k])
        events.append(current_event)

        # process events/plates
        for event_votes in events:
            total_votes = len(event_votes)

            if total_votes < min_votes:
                continue

            # find most common text
            texts = [v[0] for v in event_votes]
            counts = Counter(texts)
            raw_winner = counts.most_common(1)[0][0]


            # find best frame start
            # filter votes that actually look like the winner to get accurate timing and not lucky guesses
            supporting_votes = [
                v for v in event_votes
                if SequenceMatcher(None, v[0], raw_winner).ratio() > 0.85
            ]
            # if text formatting changed the text so much somehow that no raw vote matches, use all votes
            if not supporting_votes:
                supporting_votes = event_votes

            supporting_votes.sort(key=lambda x: x[1])
            #it fixes the evaluaiton bug where we find frames early too
            # pick a frame slightly into the sequence to ensure full visibility for evaluation .py thing
            safe_idx = min(len(supporting_votes) - 1, 2)
            best_pred = supporting_votes[safe_idx]

            best_frame = best_pred[1]
            best_time = best_pred[2]

            initial_results.append((raw_winner, best_frame, best_time, total_votes))


    if not initial_results:
        return []

    initial_results.sort(key=lambda x: x[1])
    final_deduped = [initial_results[0]]

    # check if we accidentally created two events for the same car because of a split cluster
    for i in range(1, len(initial_results)):
        curr = initial_results[i]
        prev = final_deduped[-1]

        frame_diff = curr[1] - prev[1]

        # filter if frames are very close and texts are somewhat similar
        if frame_diff < 20:
            # they are essentially the same car (text match > 0.7) maybe mess with values tho
            if SequenceMatcher(None, curr[0], prev[0]).ratio() > 0.7:
                # keep the one with more votes
                if curr[3] > prev[3]:
                    final_deduped[-1] = curr
                continue  # skip adding curr as a new event plate since we merged it

        final_deduped.append(curr)

    return final_deduped

def CaptureFrame_Process(file_path, sample_frequency, save_path):
    frames_captured = []
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # process every Nth frame (1 or 2 for high accuracy)
    frame_interval = sample_frequency
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frames_captured.append((frame, frame_count, timestamp))
        frame_count += 1

    cap.release()

    # create directories
    #output_dir = "custom_output"
    #os.makedirs(output_dir, exist_ok=True)

    # standard Setup
    #np.random.seed(42)
    #total_frames = len(frames_captured)
    #validation_size = total_frames // 3
    #indices = np.arange(total_frames)
    #np.random.shuffle(indices)
    #validation_indices = indices[:validation_size]
    #training_indices = indices[validation_size:]

    #validation_frames = [frames_captured[i] for i in validation_indices]
    #training_frames = [frames_captured[i] for i in training_indices]

    #comment in regular use
    #training_frames = frames_captured

    #regular running
    #ValidationEvaluation.validationSetup(validation_frames)
    #frames_captured = training_frames

    #test_dir = "training_set/training_frames"
    #os.makedirs(test_dir, exist_ok=True)
    #for v_frame, v_num, v_ts in training_frames:
    #    filename = os.path.join(test_dir, f"frame_{v_num:06d}.jpg")
    #    cv2.imwrite(filename, v_frame)

    #test_plate_dir = "training_set/training_output"
    #os.makedirs(test_plate_dir, exist_ok=True)

    # load char dataset for recognize module
    char_dataset_dir = 'dataset/CharsLabeled'
    char_dataset = []
    if os.path.exists(char_dataset_dir):
        for char_file in os.listdir(char_dataset_dir):
            path = os.path.join(char_dataset_dir, char_file)
            # check for valid image file
            if os.path.isfile(path) and not char_file.startswith('.'):
                img = cv2.imread(path)
                if img is not None:
                    char = char_file[0]
                    char_dataset.append((char, img[:, :, 0]))

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds),Votes\n")

    # load the file
    #df = pd.read_csv('dataset/groundTruth.csv')

    #create a set of valid license plates for quick checking
    #ground_truth_plates = set(df['License plate'].unique())

    # voting variables
    session_buffer = []
    last_plate_timestamp = -1.0

    # if gap > threshold assumes major batch of video done
    gap_threshold_seconds = 2

    correct = 0
    min_votes = 100
    for fr in frames_captured:
        current_frame_image = np.array(fr[0])
        current_frame_num = fr[1]
        current_timestamp = fr[2]

        # optional save frame for debugging
        # filename = os.path.join(training_dir, f"frame_{current_frame_num:06d}.jpg")
        # cv2.imwrite(filename, current_frame_image)

        # gap checker -> flush logic for plates detected
        if last_plate_timestamp != -1.0 and (current_timestamp - last_plate_timestamp > gap_threshold_seconds):
            best_list = get_best_predictions_clustering(session_buffer,fps)
            for best in best_list:
                # best = (plate, frame, time, votes)
                #suffix = ""
                #if best[0] in ground_truth_plates:
                  #  suffix = " !!correct!!"
                 #   correct += 1
                 #   if best[3] < min_votes:
                #        min_votes = best[3]
                #else:
                #    suffix = " XXXX"
                #print(f"Car Left. Writing: {best[0]} ({best[3]} votes){suffix} frame num: {best[1]} timestamp: {best[2]}", flush=True)
                output.write(f"{best[0]},{best[1]},{best[2]},{best[3]}\n")
            session_buffer = []
            last_plate_timestamp = -1.0

        # plate localisation
        processed_plates = Localization.plate_detection(current_frame_image)
        found_plate_in_this_frame = False

        for i, plate in enumerate(processed_plates):
            # optional save cropped plate
            # filename = os.path.join(output_dir, f"frame_{current_frame_num:06d}_{i}.jpg")
            # cv2.imwrite(filename, plate['img'])
            #filename = os.path.join(test_plate_dir, f"frame_{current_frame_num:06d}_{i}.jpg")
            #cv2.imwrite(filename, plate['img'])

            # recognition
            recognized_plate_num = Recognize.segment_and_recognize(
                plate['img'],
                current_frame_num,
                char_dataset
                #,thresh_method='adaptive_mean',
                #blk=15,
                #c=5
            )

            if recognized_plate_num and len(recognized_plate_num) > 1 and recognized_plate_num.strip('-'):
                # store prediction in buffer
                session_buffer.append((recognized_plate_num, current_frame_num, current_timestamp))
                found_plate_in_this_frame = True

        if found_plate_in_this_frame:
            last_plate_timestamp = current_timestamp

    # final flush for end of video
    if session_buffer:
        #change it so it doesnt cancel and add so many frames togeter cause if there is more than 120 frames between they are different
        best_list = get_best_predictions_clustering(session_buffer,fps)
        for best in best_list:
            #print(f"End of Video. Writing: {best[0]} ({best[3]} votes)", flush=True)
            output.write(f"{best[0]},{best[1]},{best[2]},{best[3]}\n")

    output.close()
    print("Capture process complete.")
    #print(f"Minimum votes for a correct plate is {min_votes}")
    #print(f"Found {correct} correct plates")
