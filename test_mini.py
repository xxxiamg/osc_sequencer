from geometry import detect_perspective
from color_analysis import dominant_colors_kmeans
from texture import texture_to_adsr
from geometry import detect_geometry
import cv2
import numpy as np
from collections import Counter

print("1. Импорты OK")
img = cv2.imread("88.jpg")
print("2. Загружено:", img.shape)

h, w = img.shape[:2]
cell = img[0:h//4, 0:w//4]
print("3. Ячейка:", cell.shape)

print("4. detect_geometry старт...")
wf, dep, stats = detect_geometry(cell)
print("5. waveform:", wf, "depth:", dep)

print("6. texture_to_adsr старт...")
a, s, r, emp, comp = texture_to_adsr(cell, (100, 100, 100))
print("7. ADSR:", a, s, r, emp, comp)

print("8. dominant_colors_kmeans старт...")
cols = dominant_colors_kmeans(cell)
print("9. Цветов:", len(cols))

print("9b. detect_perspective старт...")
persp = detect_perspective(img)
print("9c. perspective:", persp)

print("10. ВСЁ ОК – скрипт завершён!")
