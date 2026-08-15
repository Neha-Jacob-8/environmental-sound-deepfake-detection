"""Generator ID mapping for EnvSDD.

The HuggingFace dataset has no G01-G07 column. The challenge generator IDs are
recovered from the composite key (attack_type, generative_model).

Verified against the split value counts: train resolves to exactly G01-G04
(6,419 reals -> 25,676 fakes, 4:1) and test to exactly G01-G07 (3,551 reals ->
24,857 fakes, 7:1).

Note G07 is AudioLDM 2 in audio-to-audio mode while G02 is the same model in
text-to-audio mode, so G07 tests an unseen *conditioning mode* for a seen
architecture. Only G05 (AudioLCM) and G06 (TangoFlux) are unseen architectures.
"""

# (attack_type, generative_model) -> generator id
GENERATOR_MAP = {
    ("tta", "audioldm1"): "G01",
    ("tta", "audioldm2"): "G02",
    ("tta", "audiogen"): "G03",
    ("ata", "audioldm1"): "G04",
    ("tta", "audiolcm"): "G05",
    ("tta", "tangoflux"): "G06",
    ("ata", "audioldm2"): "G07",
    ("real", "real"): "REAL",
}

SEEN_GENERATORS = ["G01", "G02", "G03", "G04"]
UNSEEN_GENERATORS = ["G05", "G06", "G07"]
ALL_GENERATORS = SEEN_GENERATORS + UNSEEN_GENERATORS

GENERATOR_NAMES = {
    "G01": "AudioLDM (TTA)",
    "G02": "AudioLDM 2 (TTA)",
    "G03": "AudioGen (TTA)",
    "G04": "AudioLDM (ATA)",
    "G05": "AudioLCM (TTA)",
    "G06": "TangoFlux (TTA)",
    "G07": "AudioLDM 2 (ATA)",
    "REAL": "Real recording",
}


def resolve_generator(attack_type, generative_model):
    """Map a row's metadata to a generator ID.

    Raises on unknown combinations rather than silently bucketing them, so a
    schema change upstream surfaces immediately instead of corrupting labels.
    """
    key = (str(attack_type).strip().lower(), str(generative_model).strip().lower())
    if key not in GENERATOR_MAP:
        raise KeyError(
            f"Unknown (attack_type, generative_model) pair: {key}. "
            f"Known pairs: {sorted(GENERATOR_MAP)}"
        )
    return GENERATOR_MAP[key]
