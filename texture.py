import cv2
import numpy as np

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
