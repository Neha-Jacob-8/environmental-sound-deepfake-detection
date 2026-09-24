"""Waveform augmentations for the generalisation experiment.

These are not here to fight overfitting - train and validation accuracy sit
within 0.03pp of each other, so there is no train/val gap to close. They target
a different failure: the detector learned "fake means far from real in one
particular direction", and generators it never trained on only travel part of
the way along it.

The hypothesis is that some of that direction is carried by *superficial* cues
which happen to correlate with the four training generators - a bandwidth
ceiling, a noise floor, an overall level - rather than by synthesis artifacts
that any generator would leave. Randomising exactly those cues should force the
model onto evidence that transfers.

Each transform therefore destroys one specific shortcut:

    gain          absolute level
    noise         the noise floor, and how clean a clip is
    lowpass       the bandwidth ceiling (G03/AudioGen has a hard one at ~6 kHz,
                  visible as a black band in its spectrogram)
    time_shift    absolute position of events in the clip

Applied to the training split only, and to real and fake clips alike: applying
them to fakes only would add a new artifact that marks the fake class, which is
the opposite of the point.
"""

import numpy as np

SR = 16_000


def gain(wav, rng, db_range=(-6.0, 6.0)):
    return wav * float(10 ** (rng.uniform(*db_range) / 20.0))


def noise(wav, rng, snr_db_range=(20.0, 45.0)):
    """Additive Gaussian noise at a random SNR."""
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 1e-8:
        return wav
    snr = rng.uniform(*snr_db_range)
    n_rms = rms / (10 ** (snr / 20.0))
    return wav + rng.normal(0.0, n_rms, size=wav.shape).astype(wav.dtype)


def lowpass(wav, rng, cutoff_range=(3500.0, 7800.0)):
    """Zero out everything above a random cutoff, via rFFT.

    A brick wall rather than a gentle filter, because the artifact being masked
    is itself a brick wall: AudioGen's output stops dead at its codec ceiling.
    """
    cutoff = rng.uniform(*cutoff_range)
    spec = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(len(wav), 1.0 / SR)
    spec[freqs > cutoff] = 0.0
    return np.fft.irfft(spec, n=len(wav)).astype(np.float32)


def time_shift(wav, rng, max_frac=0.5):
    """Circular shift, so the clip stays exactly 64,000 samples."""
    return np.roll(wav, int(rng.uniform(-max_frac, max_frac) * len(wav)))


TRANSFORMS = {"gain": gain, "noise": noise, "lowpass": lowpass,
              "time_shift": time_shift}
DEFAULT = ("gain", "noise", "lowpass", "time_shift")


class Augment:
    """Applies each enabled transform independently with probability `p`.

    Args:
        transforms: names from TRANSFORMS; defaults to all four.
        p: per-transform probability, so a clip may get none, some or all.
        seed: augmentation draws come from their own generator, kept separate
            from the training seed so changing one does not silently change the
            other.
    """

    def __init__(self, transforms=DEFAULT, p=0.5, seed=0):
        unknown = set(transforms) - set(TRANSFORMS)
        if unknown:
            raise ValueError(f"unknown transform(s): {sorted(unknown)}. "
                             f"Available: {sorted(TRANSFORMS)}")
        self.names = list(transforms)
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def __call__(self, wav):
        out = wav
        for name in self.names:
            if self.rng.random() < self.p:
                out = TRANSFORMS[name](out, self.rng)
        return np.asarray(out, dtype=np.float32)

    def __repr__(self):
        return f"Augment({'+'.join(self.names)}, p={self.p})"
