import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "speed_skydive.db"
RAW_UPLOAD_DIR = BASE_DIR / "uploaded_logs"

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


def _load_local_env() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = raw_value.strip().strip('"').strip("'")
        os.environ[key] = value


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip().replace(",", "."))
    except ValueError:
        return default


_load_local_env()

COACH_VIEW_ENABLED = _env_flag("COACH_VIEW_ENABLED", False)
AI_COACHING_ENABLED = _env_flag("AI_COACHING_ENABLED", False)
AI_COACHING_MODEL = os.getenv("AI_COACHING_MODEL", "gpt-5.5").strip() or "gpt-5.5"
AI_COACHING_TIMEOUT_S = _env_float("AI_COACHING_TIMEOUT_S", 12.0)
