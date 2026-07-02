import numpy as np

def pitch_shift(y, cents):
    """Resample audio slightly to pitch shift without complex dependencies."""
    factor = 2 ** (cents / 1200.0)
    n_samples = int(len(y) / factor)
    x = np.arange(len(y))
    return np.interp(np.linspace(0, len(y) - 1, n_samples), x, y).astype(np.float32)

def apply_chorus(audio, sr):
    """Simulate group chanting by mixing two delay-pitch shifted copies of the audio."""
    # Copy 1: delay by 18ms, pitch shift +6 cents
    delay1_samples = int(0.018 * sr)
    shifted1 = pitch_shift(audio, 6)
    c1 = np.zeros_like(audio)
    if len(shifted1) > delay1_samples:
        c1[delay1_samples:] = shifted1[:len(audio) - delay1_samples]
        
    # Copy 2: delay by 28ms, pitch shift -6 cents
    delay2_samples = int(0.028 * sr)
    shifted2 = pitch_shift(audio, -6)
    c2 = np.zeros_like(audio)
    if len(shifted2) > delay2_samples:
        c2[delay2_samples:] = shifted2[:len(audio) - delay2_samples]
        
    # Mix original and copies
    mixed = audio * 0.55 + c1 * 0.25 + c2 * 0.20
    mx = np.abs(mixed).max()
    if mx > 1.0:
        mixed = mixed / mx * 0.97
    return mixed

def apply_reverb(audio, sr, room_type="temple"):
    """Implement a Schroeder reverberator (parallel combs followed by series all-passes)."""
    if room_type == "temple":
        comb_delays = [0.035, 0.041, 0.047, 0.053]
        comb_gains = [0.75, 0.72, 0.70, 0.68]
        allpass_delays = [0.005, 0.0017]
        allpass_gains = [0.7, 0.7]
        dry_wet = 0.28
    elif room_type == "cave":
        comb_delays = [0.050, 0.056, 0.062, 0.068]
        comb_gains = [0.82, 0.80, 0.78, 0.76]
        allpass_delays = [0.008, 0.003]
        allpass_gains = [0.7, 0.7]
        dry_wet = 0.38
    else:  # studio
        comb_delays = [0.018, 0.022, 0.026, 0.030]
        comb_gains = [0.42, 0.40, 0.38, 0.36]
        allpass_delays = [0.003, 0.001]
        allpass_gains = [0.5, 0.5]
        dry_wet = 0.12

    # Fast iterative comb filter
    def comb_filter(x, delay_s, g):
        d = int(delay_s * sr)
        y = np.copy(x)
        for n in range(d, len(x)):
            y[n] = x[n] + g * y[n - d]
        return y

    # Fast iterative all-pass filter
    def allpass_filter(x, delay_s, g):
        d = int(delay_s * sr)
        y = np.zeros_like(x)
        for n in range(len(x)):
            if n >= d:
                y[n] = -g * x[n] + x[n - d] + g * y[n - d]
            else:
                y[n] = -g * x[n]
        return y

    combs = [comb_filter(audio, d, g) for d, g in zip(comb_delays, comb_gains)]
    wet = np.sum(combs, axis=0) / len(combs)

    for d, g in zip(allpass_delays, allpass_gains):
        wet = allpass_filter(wet, d, g)

    mixed = (1.0 - dry_wet) * audio + dry_wet * wet
    mx = np.abs(mixed).max()
    if mx > 1.0:
        mixed = mixed / mx * 0.97
    return mixed

def synthesize_drone(duration, sr, key="C#"):
    """Synthesize a meditative Tanpura-like string drone with slow LFOs and organic harmonics."""
    keys = {
        "C#": (138.59, 207.65),
        "D": (146.83, 220.00),
        "D#": (155.56, 233.08),
        "E": (164.81, 246.94),
        "F": (174.61, 261.63),
        "F#": (185.00, 277.18),
        "G": (196.00, 293.66),
        "G#": (207.65, 311.13),
        "A": (220.00, 329.63),
        "A#": (233.08, 349.23),
        "B": (246.94, 369.99),
        "C": (261.63, 392.00)
    }
    sa, pa = keys.get(key.upper().strip(), (138.59, 207.65))
    t = np.arange(int(duration * sr)) / float(sr)
    
    # Pluck/vibration modulation (slow LFOs)
    lfo1 = 0.75 + 0.25 * np.sin(2 * np.pi * 0.18 * t)
    lfo2 = 0.75 + 0.25 * np.sin(2 * np.pi * 0.32 * t)
    
    drone = (
        0.35 * np.sin(2 * np.pi * sa * t) * lfo1 +
        0.20 * np.sin(2 * np.pi * sa * 2 * t) * lfo1 +
        0.10 * np.sin(2 * np.pi * sa * 3 * t) * lfo1 +
        0.05 * np.sin(2 * np.pi * sa * 4 * t) * lfo1 +
        0.25 * np.sin(2 * np.pi * pa * t) * lfo2 +
        0.12 * np.sin(2 * np.pi * pa * 2 * t) * lfo2 +
        0.20 * np.sin(2 * np.pi * (sa * 2) * t) * lfo1
    )
    
    # Fade in / out to make it smooth
    fade_len = int(0.5 * sr)
    if len(drone) > 2 * fade_len:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        drone[:fade_len] *= fade_in
        drone[-fade_len:] *= fade_out

    mx = np.abs(drone).max()
    if mx > 0:
        drone = drone / mx
    return drone.astype(np.float32)

def mix_drone(audio, drone, volume_db=-24):
    """Mix generated audio with the background drone at a specific decibel level."""
    factor = 10 ** (volume_db / 20.0)
    # Ensure drone matches length of audio
    if len(drone) > len(audio):
        drone_aligned = drone[:len(audio)]
    else:
        drone_aligned = np.pad(drone, (0, len(audio) - len(drone)), mode='wrap')
        
    mixed = audio + drone_aligned * factor
    mx = np.abs(mixed).max()
    if mx > 1.0:
        mixed = mixed / mx * 0.97
    return mixed
