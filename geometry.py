import cv2
import numpy as np
from sklearn.decomposition import PCA


def classify_lines(edges, gray):
    """Анализирует линии и контуры, возвращает словарь с количеством
    горизонтальных, вертикальных, диагональных и кривых элементов."""
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10,
                            minLineLength=5, maxLineGap=3)
    horizontal = vertical = diagonal = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) % 180
            if angle < 10 or angle > 170:
                horizontal += 1
            elif 80 < angle < 100:
                vertical += 1
            elif (10 <= angle <= 80) or (100 <= angle <= 170):
                diagonal += 1

    # Кривые через PCA контуров
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curves = 0
    for cnt in contours:
        if len(cnt) < 5:
            continue
        pts = cnt.reshape(-1, 2)
        if pts.shape[0] < 5:
            continue
        pca = PCA(n_components=2)
        pca.fit(pts)
        # не вытянутая форма => кривая
        if pca.explained_variance_ratio_[0] < 0.8:
            curves += 1
    return {'horizontal': horizontal, 'vertical': vertical,
            'diagonal': diagonal, 'curves': curves}


def detect_geometry(cell_bgr):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 50, 150)

    # print(".", end="", flush=True)   # отладка

    line_stats = classify_lines(edges, gray)

    num_lines = sum([line_stats['horizontal'], line_stats['vertical'],
                     line_stats['diagonal']])
    num_curves = line_stats['curves']

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                               param1=50, param2=15, minRadius=2, maxRadius=20)
    num_circles = len(circles[0]) if circles is not None else 0

    edge_ratio = np.count_nonzero(edges) / edges.size

    # Определяем форму волны с учётом типов линий
    if num_circles > 3:
        waveform = 'sine'
    elif line_stats['curves'] > 2:
        waveform = 'sine'      # кривые -> синус
    elif line_stats['diagonal'] > line_stats['horizontal'] + line_stats['vertical']:
        waveform = 'saw'       # диагонали -> пила
    elif line_stats['vertical'] > 2 * line_stats['horizontal']:
        waveform = 'ramp'      # вертикальные -> рампа
    elif line_stats['horizontal'] > 2 * line_stats['vertical']:
        waveform = 'pulse'     # горизонтальные -> импульс
    elif num_lines > 5:
        waveform = 'square'
    elif edge_ratio > 0.2:
        waveform = 'noise'
    else:
        waveform = 'triangle'

    # Глубина модуляции на основе угловых точек
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=20,
                                      qualityLevel=0.01, minDistance=3)
    num_corners = len(corners) if corners is not None else 0
    mod_depth = min(1.0, num_corners / 15.0)

    return waveform, mod_depth, line_stats


def detect_perspective(image_bgr):
    """Быстрая оценка перспективы по уменьшенному изображению."""
    h, w = image_bgr.shape[:2]
    # Работаем с уменьшенной копией
    scale = min(1.0, 400.0 / max(w, h))
    small = cv2.resize(image_bgr, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    if lines is None:
        return {'depth': 0.0, 'width': 0.0, 'pan': 0.0}

    # Ограничиваем количество линий для быстродействия
    max_lines = 100
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    intersections = []
    for i in range(len(lines)):
        rho1, theta1 = lines[i][0]
        A1 = np.array([[np.cos(theta1), np.sin(theta1)]])
        for j in range(i+1, len(lines)):
            rho2, theta2 = lines[j][0]
            A2 = np.array([[np.cos(theta2), np.sin(theta2)]])
            A = np.vstack([A1, A2])
            b = np.array([rho1, rho2])
            if np.linalg.matrix_rank(A) < 2:
                continue
            pt = np.linalg.lstsq(A, b, rcond=None)[0]
            if 0 <= pt[0] < small.shape[1] and 0 <= pt[1] < small.shape[0]:
                pt_orig = pt / scale
                intersections.append(pt_orig)

    if not intersections:
        return {'depth': 0.0, 'width': 0.0, 'pan': 0.0}

    pts = np.array(intersections)
    center = np.array([w/2, h/2])
    vanish_point = np.median(pts, axis=0)
    depth = 1.0 - np.linalg.norm(vanish_point - center) / max(h, w)
    width = np.std(pts[:, 0]) / w
    pan = (vanish_point[0] - center[0]) / (w/2)
    return {'depth': round(depth, 3), 'width': round(width, 3), 'pan': round(pan, 3)}
