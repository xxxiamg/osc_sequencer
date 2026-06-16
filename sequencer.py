import cv2
import numpy as np
from collections import Counter
from tqdm import tqdm

from geometry import detect_geometry, detect_perspective
from texture import texture_to_adsr
from color_analysis import dominant_colors_kmeans, find_lead_object_mask, determine_color_harmony
from mappings import hue_to_midi, saturation_to_amp, brightness_to_octave


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
    perspective = detect_perspective(img)

    cell_h = h // grid_size
    cell_w = w // grid_size
    steps = []

    total_cells = grid_size * grid_size
    with tqdm(total=total_cells, desc="Анализ ячеек", unit="cell") as pbar:
        for row in range(grid_size):
            for col in range(grid_size):
                y1 = row * cell_h
                y2 = y1 + cell_h
                x1 = col * cell_w
                x2 = x1 + cell_w
                cell_bgr = img[y1:y2, x1:x2]
                if cell_bgr.size == 0:
                    pbar.update(1)
                    continue

                step_index = row * grid_size + col

                waveform, fm_depth, line_stats = detect_geometry(cell_bgr)
                attack, sustain, release, emptiness, complexity = texture_to_adsr(
                    cell_bgr, bg_color)

                cell_lead = lead_mask[y1:y2, x1:x2]
                lead_ratio = np.count_nonzero(cell_lead) / cell_lead.size
                is_lead = lead_ratio > 0.3

                dom_colors = dominant_colors_kmeans(cell_bgr, k=3)
                voices = []
                hue_list = []
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
                    hue_list.append(hue)

                harmony_type, harmony_shift = determine_color_harmony(hue_list)

                gate = 1.0 - emptiness
                gate = np.clip(gate, 0.0, 1.0)

                balance = 0.0
                symmetry = 0.0
                density = lead_ratio
                if cell_lead.size > 0:
                    h_cell, w_cell = cell_lead.shape
                    thirds_h = h_cell // 3
                    top = np.count_nonzero(cell_lead[:thirds_h, :])
                    mid = np.count_nonzero(cell_lead[thirds_h:2*thirds_h, :])
                    bot = np.count_nonzero(cell_lead[2*thirds_h:, :])
                    total = top + mid + bot + 1e-7
                    balance = (bot - top) / total
                    symmetry = 1.0 - abs(top - bot) / total

                step = {
                    'step': step_index,
                    'waveform': waveform,
                    'modulation_depth': round(fm_depth, 3),
                    'attack': round(attack, 3),
                    'sustain': round(sustain, 3),
                    'release': round(release, 3),
                    'is_lead': is_lead,
                    'voices': voices,
                    'gate': round(gate, 3),
                    'emptiness': round(emptiness, 3),
                    'complexity': round(complexity, 3),
                    'harmony_type': harmony_type,
                    'harmony_shift': harmony_shift,
                    'balance': round(balance, 3),
                    'symmetry': round(symmetry, 3),
                    'density': round(density, 3),
                    'line_horizontal': line_stats['horizontal'],
                    'line_vertical': line_stats['vertical'],
                    'line_diagonal': line_stats['diagonal'],
                    'line_curves': line_stats['curves']
                }
                steps.append(step)
                pbar.update(1)

    for step in steps:
        step['scene_id'] = 0
        step['perspective_pan'] = perspective['pan']
        step['perspective_depth'] = perspective['depth']
        step['perspective_width'] = perspective['width']

    return steps
