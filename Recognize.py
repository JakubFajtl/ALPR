import cv2
import numpy as np


# --- MAIN PIPELINE ---

def prepare_plate(plate_image, method='otsu', block_size=11, c_value=9):
    """
    Prepares plate with configurable thresholding.
    method: 'otsu' (Default), 'adaptive_mean', 'adaptive_gauss', 'manual'
    """
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image

    target_height = 80
    aspect_ratio = gray.shape[1] / float(gray.shape[0])
    new_width = int(target_height * aspect_ratio)
    img = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

    # force block size
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    # thresholding logic
    if method == 'otsu':
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elif method == 'adaptive_mean':
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, c_value)

    elif method == 'adaptive_gauss':
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, c_value)

    elif method == 'manual':
        _, binary = cv2.threshold(img, c_value, 255, cv2.THRESH_BINARY_INV)

    else:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cleaning - morph + contours
    cleaned = clean_noise(binary)
    cleaned = morphological_cleanup(cleaned)
    final = clean_plate_by_contours(cleaned)
    return final

def segment_and_recognize(plate_image, frame_num, char_dataset, thresh_method='otsu', blk=11, c=9):
    # configuration list for multipass strategy
    configs = [
        (thresh_method, blk, c),  # pass 1 user / default
        ('adaptive_mean', 13, 10),  # pass 2 mean adaptive
        ('adaptive_gauss', 15, 8),  # pass 3 gaussian
        ('manual', 6, 81),  # detect some edge cases
        ('otsu', 11, 9)  # pass 4 fallback
    ]

    best_result = "?"
    lowest_cost = float('inf')

    # iterate through configurations
    for method, block, c_val in configs:
        cleaned_plate = prepare_plate(plate_image, method=method, block_size=block, c_value=c_val)
        chars, _ = segment_characters_by_projection(cleaned_plate)
        plate_height = cleaned_plate.shape[0]

        raw_result = read_characters(chars, char_dataset, plate_height)
        formatted_plate = smart_format_dutch_plate(raw_result)

        if formatted_plate is None or formatted_plate == "?" or formatted_plate == "":
            current_cost = 999
        else:
            current_cost = calculate_pattern_match_cost(formatted_plate)
        # return immediately on perfect match
        if current_cost == 0:
            return formatted_plate

        if current_cost < lowest_cost:
            lowest_cost = current_cost
            best_result = formatted_plate

    # fallback if no perfect match found
    if best_result == "?" or best_result is None:
        cleaned_plate = prepare_plate(plate_image, method='otsu')
        chars, _ = segment_characters_by_projection(cleaned_plate)
        return read_characters(chars, char_dataset, 80)

    #visualize_plate_chars(chars, plate_height)
    return best_result

def calculate_pattern_match_cost(plate_str):
    """Returns 0 for perfect match and more than 0 for imperfections."""
    patterns = ["NN-LLL-N", "N-LLL-NN", "NN-LL-LL", "LL-NNN-L",
                "LL-LL-NN", "LL-NN-NN", "NN-NN-LL", "L-NNN-LL", "LL-LLL-N"]

    clean = plate_str.replace('-', '')
    if len(clean) != 6: return 50  # wrong length penalty

    # check if structure matches any pattern perfectly
    for pat in patterns:
        flat_pat = pat.replace('-', '')
        match = True
        for char, type_req in zip(clean, flat_pat):
            if type_req == 'N' and not char.isdigit(): match = False
            if type_req == 'L' and not char.isalpha(): match = False

        # check dash positions
        formatted_dashes = [i for i, c in enumerate(plate_str) if c == '-']
        pat_dashes = [i for i, c in enumerate(pat) if c == '-']

        if match and formatted_dashes == pat_dashes:
            return 0  # perfect

    return 10  # imperfect match

def read_characters(chars, char_dataset, plate_height=80):
    recognized_plate = ""
    for char in chars:
        # remove top/bottom noise
        clean_char = clean_single_char(char, plate_height)
        # if cleaning removed everything skip it
        if clean_char is None:
            continue
        recognized_plate += read_char(crop_character(clean_char), char_dataset, recognized_plate, plate_height)
    return recognized_plate

def read_char(char, char_dataset, history, plate_height=80):
    if char.size == 0: return "?"

    # predict this chars expected type n/l
    expected_type = predict_next_type(history)

    # geometric dash check
    if is_geometric_dash(char, plate_height):
        if expected_type in ['N', 'L'] and not is_strong_dash(char):
            pass
        else:
            return "-"

    # template matchin part
    best_char = '?'
    min_score = float('inf')

    for character_symbol, reference_character in char_dataset:
        ref_h, ref_w = reference_character.shape
        char_h, char_w = char.shape

        scale = ref_h / char_h
        new_width = int(char_w * scale)

        # discard if too thin
        if new_width < 4:
            return ""

        if new_width <= 0: continue

        resized = cv2.resize(char, (new_width, ref_h), interpolation=cv2.INTER_CUBIC)
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        if new_width > ref_w:
            candidate = resized[:, :ref_w]
        elif new_width < ref_w:
            padding = ref_w - new_width
            candidate = np.pad(resized, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        else:
            candidate = resized

        # jitter match
        jitter_range = 3
        padded_ref = cv2.copyMakeBorder(reference_character, jitter_range, jitter_range, jitter_range, jitter_range,
                                        cv2.BORDER_CONSTANT, value=0)

        best_local_score = float('inf')
        for dy in range(0, 2 * jitter_range + 1, 2):
            for dx in range(0, 2 * jitter_range + 1, 2):
                ref_window = padded_ref[dy: dy + ref_h, dx: dx + ref_w]
                xor_diff = cv2.bitwise_xor(candidate, ref_window)
                diff_count = cv2.countNonZero(xor_diff)
                if diff_count < best_local_score:
                    best_local_score = diff_count

        final_score = best_local_score / (ref_h * ref_w)

        if final_score < min_score:
            min_score = final_score
            best_char = character_symbol
    # disambiguation
    if best_char in ['T', '1', 'J']:
        return disambiguate_J_T_1(char, expected_type)
    if best_char in ['Z', '2']:
        return disambiguate_Z_2(char, expected_type)
    if best_char in ['B', '8']:
        return disambiguate_B_8(char, expected_type)
    if best_char in ['S', '5']:
        return disambiguate_S_5(char, expected_type)
    if best_char in ['4', '9']:
        return disambiguate_4_9(char)

    return best_char

# --- PREDICTIVE LOGIC ---

def predict_next_type(history):
    if not history: return None
    patterns = ["NN-LLL-N", "N-LLL-NN", "NN-LL-LL", "LL-NNN-L", "LL-LL-NN", "LL-NN-NN", "NN-NN-LL", "L-NNN-LL",
                "LL-LLL-N", "LLL-NN-L", "L-NN-LLL", "N-LL-NNN",
                "NNN-LL-N"]  # added newer plate styles incase new video used
    possible_next_types = set()
    current_idx = len(history)

    for pat in patterns:
        if is_prefix_match(history, pat):
            if current_idx < len(pat):
                possible_next_types.add(pat[current_idx])

    if len(possible_next_types) == 1:
        return list(possible_next_types)[0]
    return None

def is_prefix_match(history, pattern):
    if len(history) > len(pattern): return False
    for i, char in enumerate(history):
        pat_char = pattern[i]
        if pat_char == '-':
            if char != '-': return False
        elif pat_char == 'N':
            if char == '-': return False
        elif pat_char == 'L':
            if char == '-': return False
    return True

def smart_format_dutch_plate(raw_text):
    """
    Smart formatting of plates based on heuristics and pattern matching
    Returns a value which is best guess / original raw format
    """
    if raw_text is None: return "?"

    candidates = [raw_text]
    ghost_dashes = ['L', 'J', '1', 'I']

    # ghost dash logic
    # general 8 chars (XX-XX-XX or similar from templates with dashes replaced by noise)
    if len(raw_text) == 8:
        temp_list = list(raw_text)
        modified = False
        for idx in [2, 5, 6]:
            if idx < len(raw_text) and raw_text[idx] in ghost_dashes:
                temp_list[idx] = '-'
                modified = True
        if modified:
            candidates.append("".join(temp_list))

    # edge cases '14LNJKL41' (length 9) but we expect '14-NJK-41'
    # the Ls at index 2 and 6 are supposed to be dashes, the original edge case plate ends with 9 not 41 but thats hadneled after
    if len(raw_text) == 9:
        temp_list = list(raw_text)
        modified = False
        # check index 2 and 6 common dash spots noise instead of dashes
        for idx in [2, 6]:
            if temp_list[idx] in ghost_dashes:
                temp_list[idx] = '-'
                modified = True

        if modified:
            # if we replaced indices 2 and 6 with dashes like in 14-NJK-41
            candidates.append("".join(temp_list))

    # noise trimming logic
    clean_raw = raw_text.replace('-', '').replace('?', '')
    if len(clean_raw) > 6:
        if len(clean_raw) == 8:
            candidates.append(clean_raw[1:-1])
        elif len(clean_raw) == 7:
            candidates.append(clean_raw[1:])
            candidates.append(clean_raw[:-1])
        elif len(clean_raw) == 9:  # handling massive noise
            candidates.append(clean_raw[1:-2])  # just a heuristic guess

    # logic for evaluation of all candidates
    best_overall_formatted = raw_text
    min_overall_cost = float('inf')

    fix_to_N = {'Z': '2', 'B': '8', 'D': '0', 'S': '5', 'T': '1', 'J': '1'}
    fix_to_L = {'2': 'Z', '8': 'B', '0': 'O', '5': 'S', '1': 'T', '4': 'A'}  # O isnt in char set and only used for some
    # trucks in NL, we added just to be safe after removing it before but i mean it cant be recognized without being in dataset
    patterns = ["NN-LLL-N", "N-LLL-NN", "NN-LL-LL", "LL-NNN-L", "LL-LL-NN", "LL-NN-NN", "NN-NN-LL", "L-NNN-LL",
                "LL-LLL-N","LLL-NN-L","L-NN-LLL","N-LL-NNN","NNN-LL-N"] #added newer plate styles incase new video used

    for candidate in candidates:
        raw_dash_indices = [i for i, c in enumerate(candidate) if c == '-']
        clean = candidate.replace('-', '').replace('?', '')

        if len(clean) < 6: continue

        best_formatted_loop = candidate
        min_cost_loop = float('inf')

        for pat in patterns:
            flat_pat = pat.replace('-', '')
            if len(clean) != len(flat_pat): continue

            current_cost = 0
            temp_corrected = []

            for char, type_req in zip(clean, flat_pat):
                is_digit = char.isdigit()
                is_alpha = char.isalpha()
                if type_req == 'N':
                    if is_digit:
                        temp_corrected.append(char)
                    elif char in fix_to_N:
                        temp_corrected.append(fix_to_N[char])
                        current_cost += 1
                    else:
                        temp_corrected.append(char)
                        current_cost += 50
                elif type_req == 'L':
                    if is_alpha:
                        temp_corrected.append(char)
                    elif char in fix_to_L:
                        temp_corrected.append(fix_to_L[char])
                        current_cost += 1
                    else:
                        temp_corrected.append(char)
                        current_cost += 50

            formatted_candidate = ""
            c_idx = 0
            pat_dash_indices = []
            for i, p_char in enumerate(pat):
                if p_char == '-':
                    formatted_candidate += '-'
                    pat_dash_indices.append(i)
                else:
                    formatted_candidate += temp_corrected[c_idx]
                    c_idx += 1

            matches = sum(1 for d in raw_dash_indices if d in pat_dash_indices)
            current_cost -= (matches * 5)

            if current_cost < min_cost_loop:
                min_cost_loop = current_cost
                best_formatted_loop = formatted_candidate

        if min_cost_loop < min_overall_cost:
            min_overall_cost = min_cost_loop
            best_overall_formatted = best_formatted_loop

    # safety check in case my logic is crazy and removes all plate guess
    if best_overall_formatted is None:
        return raw_text

    return best_overall_formatted

# --- DISAMBIGUATION HELPERS ---

def disambiguate_4_9(char_img):
    # geometric check for them
    # 4 usually has an open top left diagonal
    # 9 has a curved closed top left
    h, w = char_img.shape
    top_half = char_img[0:int(h * 0.5), :]

    # focus on top left quadrant
    tl_quad = top_half[:, 0:int(w * 0.5)]
    if tl_quad.size == 0: return '4'

    fill_ratio = cv2.countNonZero(tl_quad) / tl_quad.size

    # 9 fills the top-left curve, 4 is usually diagonal (less pixels in top-left corner)
    if fill_ratio > 0.45:
        return '9'
    else:
        return '4'

def disambiguate_J_T_1(char_img, hint=None):
    if hint == 'N': return '1'
    h, w = char_img.shape
    bot_left = char_img[int(h * 0.75):, 0:int(w * 0.4)]
    fill_bl = cv2.countNonZero(bot_left) / bot_left.size if bot_left.size > 0 else 0
    if fill_bl > 0.35: return 'J'

    top_slice = char_img[0:int(h * 0.2), :]
    mid_slice = char_img[int(h * 0.4):int(h * 0.6), :]
    if cv2.countNonZero(mid_slice) > 0:
        ratio = cv2.countNonZero(top_slice) / cv2.countNonZero(mid_slice)
        if ratio > 1.8: return 'T'
    if hint == 'L': return 'T' # if not found still probasbly T if hint is L, either way it kinda never gets that hint
    return '1'

def disambiguate_Z_2(char_img, hint=None):
    img = cv2.resize(char_img, (32, 32))
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # calculate symmetry score
    rotated = cv2.rotate(binary, cv2.ROTATE_180)
    diff = cv2.absdiff(binary, rotated)

    pixels = cv2.countNonZero(binary)
    diff_pixels = cv2.countNonZero(diff)

    # 0.0 is perfect symmetry (Z), 1.0 is no overlap (2)
    score = diff_pixels / pixels if pixels > 0 else 0

    prediction = 'Z' if score < 0.55 else '2'

    #plt.figure(figsize=(3, 3))
    #plt.imshow(binary, cmap='gray')
    #color = 'green' if prediction == 'Z' else 'red'
    #plt.title(f"Pred: {prediction}\nScore: {score:.3f}", color=color, fontweight='bold')
    #plt.axis('off')
    #plt.show()
    return prediction

def disambiguate_B_8(char_img, hint=None):
    if hint == 'L': return 'B'
    if hint == 'N': return '8'
    h, w = char_img.shape
    mid_y = h // 2
    left_strip = char_img[mid_y - 5:mid_y + 5, 0:int(w * 0.25)]
    fill = cv2.countNonZero(left_strip) / left_strip.size if left_strip.size > 0 else 0
    return 'B' if fill > 0.6 else '8'

def disambiguate_S_5(char_img, hint=None):
    if hint == 'L': return 'S'
    if hint == 'N': return '5'
    h, w = char_img.shape
    top_bar = char_img[0:int(h * 0.12), :]
    fill = cv2.countNonZero(top_bar) / top_bar.size if top_bar.size > 0 else 0
    return '5' if fill > 0.6 else 'S'

# --- HELPERS ---

def is_geometric_dash(char_img, plate_height=80):
    h, w = char_img.shape
    if h > (plate_height * 0.35): return False
    if w > h * 1.2: return True
    contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_y = h // 2
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        blob_cy = y + (ch // 2)
        if cw > ch and abs(blob_cy - center_y) < h * 0.2:
            if cw > w * 0.5: return True
    return False

def is_strong_dash(char_img):
    h, w = char_img.shape
    return w > h * 1.5

def crop_character(char):
    if char.size == 0: return char
    rows = np.any(char, axis=1)
    cols = np.any(char, axis=0)
    if not np.any(rows) or not np.any(cols): return char
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return char[rmin:rmax + 1, cmin:cmax + 1]

# --- CLEANING UTILS ---

def morphological_cleanup(binary_img):
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_vert)
    return cleaned

def clean_noise(image):
    h, w = image.shape
    transitions = np.abs(np.diff(image, axis=1))
    row_scores = np.sum(transitions > 0, axis=1)
    min_transitions = 8
    text_rows = np.where(row_scores > min_transitions)[0]
    if len(text_rows) == 0: return image
    coords = np.column_stack((text_rows[:-1], text_rows[1:]))
    splits = np.where(coords[:, 1] - coords[:, 0] > 5)[0]
    if len(splits) > 0:
        groups = np.split(text_rows, splits + 1)
        main_group = max(groups, key=len)
    else:
        main_group = text_rows
    y_top = max(0, main_group[0] - 2)
    y_bottom = min(h, main_group[-1] + 2)
    clean_mask = image[y_top:y_bottom, :]
    return clean_mask

def clean_plate_by_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    final_plate = np.zeros_like(image)
    h, w = image.shape
    img_center_y = h // 2
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1: continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            cy = y + (ch // 2)
            min_char_height = h * 0.25
            min_char_width = w * 0.02
            dash_min_height = h * 0.03
            dash_max_height = h * 0.25
            dash_max_width = w * 0.15
            is_char = True
            if ch < min_char_height: is_char = False
            if cw < min_char_width: is_char = False
            if abs(cy - img_center_y) > 20: is_char = False
            is_dash = False
            if not is_char:
                if dash_min_height < ch < dash_max_height:
                    if abs(cy - img_center_y) < h * 0.25:
                        if cw > ch and cw < dash_max_width:
                            is_dash = True
            if not is_char and not is_dash: continue
            cv2.drawContours(final_plate, contours, i, (255), cv2.FILLED)
            inner = hierarchy[i][2]
            while inner != -1:
                cv2.drawContours(final_plate, contours, inner, (0), cv2.FILLED)
                inner = hierarchy[inner][0]
    return final_plate

def segment_characters_by_projection(clean_plate):
    h, w = clean_plate.shape

    #create a BGR copy of the plate so we can draw colored lines
    #debug_img = cv2.cvtColor(clean_plate, cv2.COLOR_GRAY2BGR)
    # Define and Draw Safe Area
    safe_top = int(h * 0.10)
    safe_bottom = int(h * 0.90)
    # draw horizontal Blue lines (BGR: 255, 0, 0)
    # cv2.line(image, start_point, end_point, color, thickness)
    #cv2.line(debug_img, (0, safe_top), (w, safe_top), (255, 0, 0), 1)
    #cv2.line(debug_img, (0, safe_bottom), (w, safe_bottom), (255, 0, 0), 1)
    # vertical projection Logic
    safe_slice = clean_plate[safe_top:safe_bottom, :]
    projection = np.sum(safe_slice, axis=0) / 255
    segments = []
    in_char = False
    start = 0
    noise_threshold = 0

    # threshold for two characters stuck together
    # if a blob is wider than 75% of the plate height, it's likely 2 chars.
    wide_threshold = int(h * 0.9) #play with value

    # helper to check aspect ratio and store segments
    def process_segment(start_x, end_x):
        char_roi = clean_plate[:, start_x:end_x]
        char_h, char_w = char_roi.shape

        if char_w > 0:
            aspect_ratio = char_w / char_h
            if char_w > 3 and aspect_ratio > 0.1:
                if char_w > wide_threshold:
                    # too wide
                    splits = split_wide_blob(char_roi)
                    segments.extend(splits)

                    # debug draw ORANGE box for split blob
                    #cv2.rectangle(debug_img, (start_x, 0), (end_x, h), (0, 165, 255), 2)
                else:
                    # normal char
                    segments.append(char_roi)

                    # debug Draw GREEN box for normal char
                    #cv2.rectangle(debug_img, (start_x, 0), (end_x, h), (0, 255, 0), 1)

    # scan the projection
    for x, count in enumerate(projection):
        has_ink = count > noise_threshold
        if has_ink and not in_char:
            in_char = True
            start = x
        elif not has_ink and in_char:
            in_char = False
            end = x
            process_segment(start, end)

    # handle the last character if it touches the right edge
    if in_char:
        process_segment(start, w)
    #debug view
    #cv2.imshow("Debug View", debug_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return segments, projection

def split_wide_blob(blob):
    # method idea - find the peak (densest part) of the halves
    h, w = blob.shape
    proj = np.sum(blob, axis=0)


    # roughly locates the center of char 1  char 2
    mid_idx = w // 2

    # search Left Peak in the first 50%
    left_region_end = max(1, mid_idx)
    peak_left_idx = np.argmax(proj[:left_region_end])

    # search Right Peak in the last 50%
    right_region_start = mid_idx
    peak_right_local = np.argmax(proj[right_region_start:])
    peak_right_idx = right_region_start + peak_right_local

    # define Cut search range (strictly between the two peaks)
    # if peaks are too close, default to middle 20-80% search
    if peak_right_idx - peak_left_idx < 5:
        search_start = int(w * 0.20)
        search_end = int(w * 0.80)
    else:
        search_start = peak_left_idx
        search_end = peak_right_idx

    # find the valley aka min in that specific range
    search_slice = proj[search_start:search_end]

    if len(search_slice) > 0:
        min_local_idx = np.argmin(search_slice)
        split_x = search_start + min_local_idx

        # verify split isn't at the very edge - helps/ed with rare edge cases
        if split_x < 5 or split_x > w - 5:
            split_x = w // 2

        return [blob[:, :split_x], blob[:, split_x:]]

    # fallback iss hard Middle Cut
    mid = w // 2
    return [blob[:, :mid], blob[:, mid:]]

def clean_single_char(char_img, plate_height):
    """
    Cleans a character by erasing noise, rather than redrawing the character.
    This preserves the internal holes of chars like 6, 8, 9, B, 4.
    """
    if char_img.size == 0: return char_img

    h, w = char_img.shape
    # find all isolated blobs (External contours)
    contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return char_img

    # find nosie blobs and earase thme
    contours_to_remove = []
    valid_candidates = []

    # define safe zone
    mid_h = h / 2
    tolerance = h * 0.30  # 30% tolerance = central 60% of image is the safe anchor

    top_limit = mid_h - tolerance  # if blob ends above this, it's top Noise
    bot_limit = mid_h + tolerance  # if blob starts below this, it's bot Noise

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        y_max = y + ch

        # check top noise - blob is in the top margin
        if y_max < top_limit:
            contours_to_remove.append(cnt)
            continue

        # check bottom noise - blob is in the bottom margin
        if y > bot_limit:
            contours_to_remove.append(cnt)
            continue

        valid_candidates.append(cnt)

    # apply eraser
    cleaned = char_img.copy()

    # draw black (0) filled polygons over the noise contours to erase them
    if contours_to_remove:
        cv2.drawContours(cleaned, contours_to_remove, -1, 0, thickness=cv2.FILLED)

    return cleaned

# --- DEBUG STUFF ---
def visualize_plate_chars(raw_chars, plate_height=80):
    """
    Debug tool: Shows all characters of a plate in one window.
    Top Row: Raw Character.
    Bottom Row: Visualization of what gets Erased (Red) vs Kept (Green).
    """
    if not raw_chars: return

    vis_h = 100
    row_raw = []
    row_debug = []

    for char in raw_chars:
        # raw version
        h, w = char.shape
        if w == 0: continue
        scale = vis_h / h
        new_w = int(w * scale)

        img_raw = cv2.resize(char, (new_w, vis_h), interpolation=cv2.INTER_CUBIC)
        img_raw_color = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        img_raw_color = cv2.copyMakeBorder(img_raw_color, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                           value=(255, 0, 0))  # Blue Border

        # debug version
        img_debug = debug_clean_single_char(char.copy(), plate_height)
        img_debug = cv2.resize(img_debug, (new_w, vis_h), interpolation=cv2.INTER_CUBIC)
        img_debug = cv2.copyMakeBorder(img_debug, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 255, 0))  # Green Border

        row_raw.append(img_raw_color)
        row_debug.append(img_debug)

    if not row_raw: return

    # stack rows
    full_row_raw = np.hstack(row_raw)
    full_row_debug = np.hstack(row_debug)
    full_display = np.vstack([full_row_raw, full_row_debug])

    cv2.imshow("Debug: Eraser Logic (Top=Raw, Bot=Result)", full_display)
    print("Press Space/Key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def debug_clean_single_char(char_img, plate_height):
    """
    Visualizes the specific 'Eraser' logic:
    - Draws Blue Lines for the safe zone.
    - Draws Red contours for what will be erased.
    - Draws Green contours for what is kept.
    """
    if char_img.size == 0: return char_img

    # convert to BGR so we can draw colors
    debug_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)

    h, w = char_img.shape
    contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return debug_img

    # replicate variables
    mid_h = h / 2
    tolerance = h * 0.30
    top_limit = int(mid_h - tolerance)
    bot_limit = int(mid_h + tolerance)

    # draw safe zone lines - blue
    cv2.line(debug_img, (0, top_limit), (w, top_limit), (255, 0, 0), 1)
    cv2.line(debug_img, (0, bot_limit), (w, bot_limit), (255, 0, 0), 1)

    # color code contours
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        y_max = y + ch

        is_noise = False

        if y_max < top_limit: is_noise = True
        if y > bot_limit: is_noise = True

        if is_noise:
            cv2.drawContours(debug_img, [cnt], -1, (0, 0, 255), 1)
        else:
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 1)

    return debug_img