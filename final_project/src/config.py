"""Central configuration. Every design decision that is a number lives here, so it is recorded and never hidden in a cell."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]                       # .../final_project
DATASET = PROJECT.parent / "ieee-phm-2012-data-challenge-dataset-master"
ARTIFACTS = PROJECT / "artifacts"
FIGURES = PROJECT / "figures"

FS = 25600.0                    # Hz, official (challenge document, Sec. 4.1)
N_SAMPLES = 2560                # samples per snapshot, official
SNAPSHOT_INTERVAL_S = 10.0      # official
FAILURE_G = 20.0                # official end-of-life criterion: vibration amplitude exceeds 20 g

LEARNING = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]
TEST = ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7", "Bearing2_3",
        "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3"]
CONDITION = {  # condition id -> (rpm, radial load N) - official, challenge document Sec. 3.2
    1: (1800, 4000), 2: (1650, 4200), 3: (1500, 5000)}
# Official actual RUL (seconds) at the truncation point of each test bearing - challenge document Table 3
OFFICIAL_RUL_S = {"Bearing1_3": 5730, "Bearing1_4": 339, "Bearing1_5": 1610, "Bearing1_6": 1460, "Bearing1_7": 7570,
                  "Bearing2_3": 7530, "Bearing2_4": 1390, "Bearing2_5": 3090, "Bearing2_6": 1290, "Bearing2_7": 580,
                  "Bearing3_3": 820}

# ---- modelling design (frozen before any test-bearing result was produced) ----
WINDOW = 16                      # snapshots per input sequence (160 s of history)
RUL_CAP_MIN = 120.0              # planning horizon: RUL above this is not distinguishable operationally
STAGE_CRITICAL_MIN = 20.0        # RUL <= 20 min  -> Critical
STAGE_WARNING_MIN = 60.0         # RUL <= 60 min  -> Warning, else Normal
ALARM_PERSISTENCE = 3            # consecutive windows needed for an alarm
SEEDS_FINAL = [42, 43, 44]
SEED_LOBO = 42
SEEDS_LOBO = [42, 43]
BATCH = 64
LR = 1e-3
MAX_EPOCHS = 40
PATIENCE = 6
LOSS_WEIGHTS = {"rul_output": 1.0, "risk_output": 0.5}
HUBER_DELTA = 0.1


def condition_of(bearing: str) -> int:
    return int(bearing.replace("Bearing", "").split("_")[0])


def subset_of(bearing: str) -> str:
    return "Learning_set" if bearing in LEARNING else "Full_Test_Set"


import os as _os
if _os.environ.get("BP_QUICK") == "1":          # debugging switch only: never used for reported results
    SEEDS_LOBO = [42]; SEEDS_FINAL = [42]; MAX_EPOCHS = 3; PATIENCE = 2
