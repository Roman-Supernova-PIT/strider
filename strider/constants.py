"""Physical constants for Roman PRISM observations."""

import numpy as np

WAVE_MIN = 7500.0
WAVE_MAX = 18000.0
N_WAVE = 1024
LOG_WAVE_MIN = np.log(WAVE_MIN)
LOG_WAVE_MAX = np.log(WAVE_MAX)
LOG_WAVE_RANGE = LOG_WAVE_MAX - LOG_WAVE_MIN

PHASE_NORM = 30.0
Z_NORM = 2.5

MAX_EPOCHS = 64
