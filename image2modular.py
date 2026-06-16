#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
абстрактная картина - потоковый OSC-секвенсор для VCV Rack.
Использование python3 image2modular.py 88.jpg --realtime --bpm 120 --loop
"""

import argparse
import json
import numpy as np
from pythonosc import udp_client
from sequencer import process_image
from osc_sender import stream_steps, send_all_at_once


class NumpyEncoder(json.JSONEncoder):
    """Кодировщик, преобразующий numpy‑числа в стандартные типы Python."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
    try:
        steps = process_image(args.image, grid_size=args.grid)
    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"Сгенерировано {len(steps)} шагов.")

    if args.savejson:
        json_path = args.image.rsplit('.', 1)[0] + "_modular.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(steps, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
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
