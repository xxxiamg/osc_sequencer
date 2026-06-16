import cv2
import numpy as np

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
