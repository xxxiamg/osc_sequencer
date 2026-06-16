import time

# Маппинг waveform в число для OSC
WAVEFORM_INDEX = {
    'triangle': 0,
    'sine': 1,
    'square': 2,
    'noise': 3,
    'saw': 4,
    'ramp': 5,
    'pulse': 6
}


def stream_steps(steps, client, prefix, bpm, loop=False):
    beat_duration = 60.0 / bpm
    step_duration = beat_duration / 4.0   # 1/16 нота
    print(f"Темп: {bpm} BPM, шаг: {step_duration*1000:.1f} мс")

    while True:
        for step in steps:
            # Базовые параметры голоса (берём первый голос)
            amp_int = int(step['voices'][0]['amplitude'] * 127)
            attack_int = int(step['attack'] / 5000.0 * 127)
            sustain_int = int(step['sustain'] * 127)
            release_int = int(step['release'] / 5000.0 * 127)
            waveform_int = WAVEFORM_INDEX.get(step['waveform'], 0)
            mod_depth_int = int(step['modulation_depth'] * 127)
            gate_int = int(step['gate'] * 127)
            scene_int = step.get('scene_id', 0)
            pan_int = int((step['perspective_pan'] + 1)
                          * 63.5)  # -1..1 -> 0..127

            client.send_message(f"{prefix}/voice/0/note",
                                step['voices'][0]['note'])
            client.send_message(f"{prefix}/voice/0/amp", amp_int)
            client.send_message(f"{prefix}/step/attack", attack_int)
            client.send_message(f"{prefix}/step/sustain", sustain_int)
            client.send_message(f"{prefix}/step/release", release_int)
            client.send_message(f"{prefix}/step/waveform", waveform_int)
            client.send_message(f"{prefix}/step/mod_depth", mod_depth_int)
            client.send_message(f"{prefix}/step/gate", gate_int)
            client.send_message(f"{prefix}/step/scene", scene_int)
            client.send_message(f"{prefix}/step/pan", pan_int)

            print(f"Шаг {step['step']+1}/{len(steps)}", end='\r')
            time.sleep(step_duration)

        if not loop:
            print("\nПоследовательность завершена.")
            break
        print("\nЦикл завершён. Повтор...")


def send_all_at_once(steps, client, prefix):
    for step in steps:
        amp_int = int(step['voices'][0]['amplitude'] * 127)
        attack_int = int(step['attack'] / 5000.0 * 127)
        sustain_int = int(step['sustain'] * 127)
        release_int = int(step['release'] / 5000.0 * 127)
        waveform_int = WAVEFORM_INDEX.get(step['waveform'], 0)
        mod_depth_int = int(step['modulation_depth'] * 127)
        gate_int = int(step['gate'] * 127)
        scene_int = step.get('scene_id', 0)
        pan_int = int((step['perspective_pan'] + 1) * 63.5)

        client.send_message(f"{prefix}/voice/0/note",
                            step['voices'][0]['note'])
        client.send_message(f"{prefix}/voice/0/amp", amp_int)
        client.send_message(f"{prefix}/step/attack", attack_int)
        client.send_message(f"{prefix}/step/sustain", sustain_int)
        client.send_message(f"{prefix}/step/release", release_int)
        client.send_message(f"{prefix}/step/waveform", waveform_int)
        client.send_message(f"{prefix}/step/mod_depth", mod_depth_int)
        client.send_message(f"{prefix}/step/gate", gate_int)
        client.send_message(f"{prefix}/step/scene", scene_int)
        client.send_message(f"{prefix}/step/pan", pan_int)
    print(f"Все {len(steps)} шагов отправлены единовременно.")
