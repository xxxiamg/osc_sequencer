import time

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
