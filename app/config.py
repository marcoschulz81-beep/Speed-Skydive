from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "speed_skydive.db"

REQUIRED_COLUMNS = [
    "time",
    "hMSL",
    "velN",
    "velE",
    "velD",
    "sAcc",
    "hAcc",
    "vAcc",
    "gpsFix",
    "numSV",
]

FIXPOINT_SECONDS = [10.0, 15.0, 20.0, 24.0, 28.0]

MAX_VALID_EXIT_ALTITUDE_AGL_M = 4267.2
PERFORMANCE_WINDOW_VERTICAL_DROP_M = 2256.0

MIN_SAMPLE_RATE_HZ = 5.0
MAX_SACC_MPS = 3.0
MIN_NUM_SV = 6

DEFAULT_BREAKOFF_ALTITUDE_AGL_M = 1700.0

TARGET_ANGLE_BANDS = [
    {"start_s": 0.0, "end_s": 3.0, "min_deg": 0.0, "max_deg": 60.0, "label": "neutraler Exit"},
    {"start_s": 3.0, "end_s": 8.0, "min_deg": 60.0, "max_deg": 70.0, "label": "Aufbauwinkel"},
    {"start_s": 8.0, "end_s": 20.0, "min_deg": 80.0, "max_deg": 85.0, "label": "Hauptaufbau"},
    {"start_s": 20.0, "end_s": 999.0, "min_deg": 83.0, "max_deg": 86.0, "label": "Peak-Haltebereich"},
]

