import sys
import time
import wave

import numpy as np

MODELS = ["tiny.en", "base.en", "small.en"]


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def load(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "use 16 kHz mono wav"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main(args):
    from faster_whisper import WhisperModel
    pairs = list(zip(args[::2], args[1::2]))
    print(f"{'model':10} {'WER':>7} {'RTF':>7}")
    for name in MODELS:
        m = WhisperModel(name, device="cpu", compute_type="int8")
        errs, comp, dur = [], 0.0, 0.0
        for path, ref in pairs:
            a = load(path)
            t0 = time.time()
            segs, _ = m.transcribe(a, language="en", beam_size=1)
            hyp = " ".join(s.text for s in segs)
            comp += time.time() - t0
            dur += len(a) / 16000
            errs.append(wer(ref, hyp))
        print(f"{name:10} {np.mean(errs):7.1%} {comp / dur:7.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
