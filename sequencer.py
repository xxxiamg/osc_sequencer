import cv2
import numpy as np
from collections import Counter
from geometry import detect_geometry
from texture import texture_to_adsr
from color_analysis import dominant_colors_kmeans, find_lead_object_mask
from mappings import hue_to_midi, saturation_to_amp, brightness_to_octave


# Разбор изображения на сетку NxN


def process_image(image_path, grid_size=16):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не могу загрузить изображение: {image_path}")
    h, w = img.shape[:2]
    print(f"Размер изображения: {w}x{h}, сетка: {grid_size}x{grid_size}")

    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels_sample = hsv_full[::10, ::10].reshape(-1, 3)
    raw_bg = Counter(tuple(p) for p in pixels_sample).most_common(1)[0][0]
    bg_color = (int(raw_bg[0]), int(raw_bg[1]), int(raw_bg[2]))

    lead_mask = find_lead_object_mask(img)

    cell_h = h // grid_size
    cell_w = w // grid_size
    steps = []

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            cell_bgr = img[y1:y2, x1:x2]
            if cell_bgr.size == 0:
                continue

            step_index = row * grid_size + col

            waveform, fm_depth = detect_geometry(cell_bgr)
            attack, sustain, release = texture_to_adsr(cell_bgr, bg_color)

            cell_lead = lead_mask[y1:y2, x1:x2]
            lead_ratio = np.count_nonzero(cell_lead) / cell_lead.size
            is_lead = lead_ratio > 0.3

            dom_colors = dominant_colors_kmeans(cell_bgr, k=3)
            voices = []
            for vid, (hue, sat, val, prop) in enumerate(dom_colors):
                base_note = hue_to_midi(hue)
                oct_shift = brightness_to_octave(val)
                note = base_note + oct_shift * 12
                amp = saturation_to_amp(sat) * prop
                voices.append({
                    'voice_id': vid,
                    'note': note,
                    'amplitude': round(np.clip(amp, 0.0, 1.0), 3),
                    'proportion': round(prop, 3)
                })

            steps.append({
                'step': step_index,
                'waveform': waveform,
                'modulation_depth': round(fm_depth, 3),
                'attack': round(attack, 3),
                'sustain': round(sustain, 3),
                'release': round(release, 3),
                'is_lead': is_lead,
                'voices': voices
            })

    return steps
