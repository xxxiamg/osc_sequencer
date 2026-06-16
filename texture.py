import cv2
import numpy as np


def box_counting_dim(binary_image, max_box_size=64):
    """Упрощённая box-counting размерность."""
    if binary_image.size == 0:
        return 0.0
    sizes = []
    counts = []
    for size in range(2, max_box_size):
        boxes = 0
        for y in range(0, binary_image.shape[0], size):
            for x in range(0, binary_image.shape[1], size):
                if np.any(binary_image[y:y+size, x:x+size]):
                    boxes += 1
        if boxes > 0:
            sizes.append(size)
            counts.append(boxes)
    if len(sizes) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def gradient_entropy(gray):
    """Энтропия распределения градиентов (хаос/порядок)."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag, _ = cv2.cartToPolar(grad_x, grad_y)
    hist = cv2.calcHist([mag.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy / 8.0


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
    lower = np.array([max(0, bg_color[0] - tolerance[0]),
                      max(0, bg_color[1] - tolerance[1]),
                      max(0, bg_color[2] - tolerance[2])], dtype=np.uint8)
    upper = np.array([min(179, bg_color[0] + tolerance[0]),
                      min(255, bg_color[1] + tolerance[1]),
                      min(255, bg_color[2] + tolerance[2])], dtype=np.uint8)
    bg_mask = cv2.inRange(hsv, lower, upper)
    emptiness = np.count_nonzero(bg_mask) / bg_mask.size
    release = 50 + emptiness * 2000

    # Временная заглушка, чтобы избежать зависания
    complexity = 0.3

    return attack, sustain, release, emptiness, complexity
