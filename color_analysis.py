import cv2
import numpy as np


def dominant_colors_kmeans(cell_bgr, k=3):
    # без изменений
    pixels = cell_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    total_pixels = len(labels)
    hsv_centers = cv2.cvtColor(centers.reshape(
        1, k, 3), cv2.COLOR_BGR2HSV).reshape(k, 3)
    counts = np.bincount(labels.flatten(), minlength=k)
    proportions = counts / total_pixels
    result = []
    for i, prop in enumerate(proportions):
        if prop > 0.03:
            h, s, v = hsv_centers[i]
            result.append((int(h), int(s), int(v), prop))
    result.sort(key=lambda x: x[3], reverse=True)
    return result


def find_lead_object_mask(image_bgr):
    # без изменений
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def determine_color_harmony(dom_hue_list):
    """
    Принимает список hue (0..179) доминирующих цветов.
    Возвращает тип гармонии (complementary, analogous, triad, monochrome)
    и предлагаемое смещение тональности для аккордов.
    """
    if len(dom_hue_list) < 2:
        return 'monochrome', 0

    hues = np.array(sorted(dom_hue_list))
    # Разница между соседними
    diffs = np.diff(hues)
    mean_diff = np.mean(diffs)

    if mean_diff < 15:
        harmony = 'monochrome'
        shift = 0
    elif 25 < mean_diff < 45:
        harmony = 'analogous'
        shift = int(mean_diff)
    elif 80 < mean_diff < 100:
        harmony = 'complementary'
        shift = 60  # тритон или кварта
    elif 50 < mean_diff < 70:
        harmony = 'triad'
        shift = 30
    else:
        harmony = 'complex'
        shift = int(mean_diff % 60)

    return harmony, shift
