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
