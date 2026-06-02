#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
абстрактная картина - потоковый OSC-секвенсор для VCV Rack.
Использование python3 image2modular.py 88.jpg --realtime --bpm 120 --loop
"""

import cv2
import numpy as np
import argparse
import time
import json
from collections import Counter
from pythonosc import udp_client

# Цвет - нота

HUE_NOTE_MAP = [
    (0, 10, 60),      # Красный -> C
    (11, 25, 62),     # Оранжевый -> D
    (26, 40, 64),     # Жёлтый -> E
    (41, 80, 65),     # Зелёный -> F
    (81, 120, 67),    # Голубой -> G
    (121, 140, 69),   # Синий -> A
    (141, 160, 71),   # Фиолетовый -> B
    (161, 179, 66),   # Пурпурный / F#
]


def hue_to_midi(hue):
    for lo, hi, note in HUE_NOTE_MAP:
        if lo <= hue <= hi:
            return note
    return 60


def saturation_to_amp(sat):
    return sat / 255.0


def brightness_to_octave(value):
    if value < 85:
        return -1
    elif value > 170:
        return 1
    return 0

# Геометрия - форма волны и глубина модуляции


def detect_geometry(cell_bgr):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10,
                            minLineLength=5, maxLineGap=3)
    num_lines = len(lines) if lines is not None else 0

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                               param1=50, param2=15, minRadius=2, maxRadius=20)
    num_circles = len(circles[0]) if circles is not None else 0

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=20,
                                      qualityLevel=0.01, minDistance=3)
    num_corners = len(corners) if corners is not None else 0

    edge_ratio = np.count_nonzero(edges) / edges.size

    if num_lines > 5 and num_circles < 3:
        waveform = 'square'
    elif num_circles > 3:
        waveform = 'sine'
    elif edge_ratio > 0.2:
        waveform = 'noise'
    else:
        waveform = 'triangle'

    mod_depth = min(1.0, num_corners / 15.0)
    return waveform, mod_depth

# Текстура - ADSR


def texture_to_adsr(cell_bgr, bg_color):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(1.0, lap_var / 200.0)

    if sharpness > 0.5:
        attack = 0.001
    else:
        attack = 200 + (1 - sharpness) * 800

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    sustain = np.clip(edge_density * 3, 0.1, 1.0)

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    tolerance = (20, 40, 40)
    lower = np.array([max(0, bg_color[0]-tolerance[0]),
                      max(0, bg_color[1]-tolerance[1]),
                      max(0, bg_color[2]-tolerance[2])], dtype=np.uint8)
    upper = np.array([min(179, bg_color[0]+tolerance[0]),
                      min(255, bg_color[1]+tolerance[1]),
                      min(255, bg_color[2]+tolerance[2])], dtype=np.uint8)
    bg_mask = cv2.inRange(hsv, lower, upper)
    emptiness = np.count_nonzero(bg_mask) / bg_mask.size
    release = 50 + emptiness * 2000

    return attack, sustain, release

# Доминирующие цвета (K-means)


def dominant_colors_kmeans(cell_bgr, k=3):
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

# Ведущий объект (самый большой контур)


def find_lead_object_mask(image_bgr):
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

# Потоковая отправка с темпом (для режима --realtime)


def stream_steps(steps, client, prefix, bpm, loop=False):
    beat_duration = 60.0 / bpm
    step_duration = beat_duration / 4.0   # 1/16 нота
    print(f"Темп: {bpm} BPM, шаг: {step_duration*1000:.1f} мс")

    while True:
        for step in steps:
            # Преобразуем в целые числа 0..127
            amp_int = int(step['voices'][0]['amplitude'] * 127)
            attack_int = int(step['attack'] / 5000.0 * 127)
            sustain_int = int(step['sustain'] * 127)
            release_int = int(step['release'] / 5000.0 * 127)

            client.send_message(f"{prefix}/voice/0/note",
                                step['voices'][0]['note'])
            client.send_message(f"{prefix}/voice/0/amp", amp_int)
            client.send_message(f"{prefix}/step/attack", attack_int)
            client.send_message(f"{prefix}/step/sustain", sustain_int)
            client.send_message(f"{prefix}/step/release", release_int)

            print(f"Шаг {step['step']+1}/{len(steps)}", end='\r')
            time.sleep(step_duration)

        if not loop:
            print("\nПоследовательность завершена.")
            break
        print("\nЦикл завершён. Повтор...")

# Статическая отправка всех шагов сразу (без --realtime)


def send_all_at_once(steps, client, prefix):
    for step in steps:
        idx = step['step']
        amp_int = int(step['voices'][0]['amplitude'] * 127)
        attack_int = int(step['attack'] / 5000.0 * 127)
        sustain_int = int(step['sustain'] * 127)
        release_int = int(step['release'] / 5000.0 * 127)

        client.send_message(f"{prefix}/voice/0/note",
                            step['voices'][0]['note'])
        client.send_message(f"{prefix}/voice/0/amp", amp_int)
        client.send_message(f"{prefix}/step/attack", attack_int)
        client.send_message(f"{prefix}/step/sustain", sustain_int)
        client.send_message(f"{prefix}/step/release", release_int)
    print(f"Все {len(steps)} шагов отправлены единовременно.")

# Главный блок


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image2Modular: картина -> OSC для VCV Rack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Пример: python image2modular.py myart.jpg --realtime --bpm 120 --loop"
    )
    parser.add_argument("image", help="Путь к изображению (jpg, png, ...)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="IP OSC-сервера (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9000,
                        help="Порт OSC (по умолчанию 9000)")
    parser.add_argument("--prefix", default="/modular",
                        help="Префикс OSC-адресов")
    parser.add_argument("--grid", type=int, default=16,
                        help="Размер сетки (NxN шагов). По умолчанию 16")
    parser.add_argument("--realtime", action="store_true",
                        help="Потоковая передача шагов с заданным BPM")
    parser.add_argument("--bpm", type=float, default=120.0,
                        help="Темп (ударов в минуту) для --realtime")
    parser.add_argument("--loop", action="store_true",
                        help="Зациклить воспроизведение (только с --realtime)")
    parser.add_argument("--savejson", action="store_true",
                        help="Сохранить результат анализа в JSON-файл")
    args = parser.parse_args()

    print("Анализ изображения...")
    steps = process_image(args.image, grid_size=args.grid)
    print(f"Сгенерировано {len(steps)} шагов.")

    if args.savejson:
        json_path = args.image.rsplit('.', 1)[0] + "_modular.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(steps, f, indent=2, ensure_ascii=False)
        print(f"JSON сохранён: {json_path}")

    client = udp_client.SimpleUDPClient(args.host, args.port)
    print(f"OSC-клиент: {args.host}:{args.port}")

    try:
        if args.realtime:
            stream_steps(steps, client, args.prefix, args.bpm, args.loop)
        else:
            send_all_at_once(steps, client, args.prefix)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    print("Готово.")
