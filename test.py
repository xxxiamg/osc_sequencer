#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import mido
import argparse
from collections import Counter


def stream_midi(steps, outport, bpm, loop=False):
    beat_duration = 60.0 / bpm
    step_duration = beat_duration / 4.0
    print(f"Темп: {bpm} BPM, шаг: {step_duration*1000:.1f} мс")

    while True:
        for step in steps:
            note = step['voices'][0]['note']
            velocity = int(step['voices'][0]['amplitude'] * 127)

            outport.send(mido.Message('note_on', note=note, velocity=velocity))
            print(f"Нота {note}, громкость {velocity}")
            time.sleep(step_duration * 0.8)
            outport.send(mido.Message('note_off', note=note))

        if not loop:
            print("Готово.")
            break
        print("Повтор...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="88.JPG")
    parser.add_argument("--port", default="IAC Driver Bus 1")
    parser.add_argument("--bpm", type=float, default=120)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()

    print("Анализ изображения...")
    steps = process_image(args.image, grid_size=args.grid)

    with mido.open_output(args.port) as outport:
        stream_midi(steps, outport, args.bpm, args.loop)
