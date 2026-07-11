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

TECHNICAL_PHASE_SPECS = [
    {
        "name": "Exit / Stabilisierung",
        "start_s": 0.0,
        "end_s": 3.0,
        "angle_min": 0.0,
        "angle_max": 40.0,
        "label": "ruhiger Exit",
    },
    {
        "name": "Dive-Aufbau",
        "start_s": 3.0,
        "end_s": 8.0,
        "angle_min": 60.0,
        "angle_max": 70.0,
        "label": "kontrollierter Aufbau",
    },
    {
        "name": "Hauptbeschleunigung",
        "start_s": 8.0,
        "end_s": 15.0,
        "angle_min": 75.0,
        "angle_max": 83.0,
        "label": "starker Speed-Aufbau",
    },
    {
        "name": "Hot-Zone Aufbau",
        "start_s": 15.0,
        "end_s": 22.0,
        "angle_min": 82.0,
        "angle_max": 86.0,
        "label": "stabile schnelle Linie",
    },
    {
        "name": "Max-Speed Fenster",
        "start_s": 20.0,
        "end_s": 28.0,
        "angle_min": 83.0,
        "angle_max": 87.0,
        "label": "reproduzierbares 3s-Fenster",
    },
]

TARGET_ANGLE_BANDS = [
    {
        "start_s": phase["start_s"],
        "end_s": phase["end_s"],
        "min_deg": phase["angle_min"],
        "max_deg": phase["angle_max"],
        "label": phase["label"],
        "name": phase["name"],
    }
    for phase in TECHNICAL_PHASE_SPECS
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


_load_local_env()

COACH_VIEW_ENABLED = _env_flag("COACH_VIEW_ENABLED", False)
AI_COACHING_ENABLED = _env_flag("AI_COACHING_ENABLED", False)
AI_COACHING_MODEL = os.getenv("AI_COACHING_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
AI_COACHING_TIMEOUT_S = _env_float("AI_COACHING_TIMEOUT_S", 12.0)
AI_COACHING_MAX_REQUESTS_PER_DAY = max(0, _env_int("AI_COACHING_MAX_REQUESTS_PER_DAY", 500))
AI_COACHING_INCLUDE_IDENTIFIERS = _env_flag("AI_COACHING_INCLUDE_IDENTIFIERS", False)
