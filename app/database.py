import sqlite3
from contextlib import contextmanager

from app.config import DATABASE_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_connection():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS jumps (
                jump_id TEXT PRIMARY KEY,
                jumper_name TEXT NOT NULL,
                file_name TEXT NOT NULL,
                device_type TEXT NOT NULL,
                raw_start_time_utc TEXT NOT NULL,
                t0_utc TEXT NOT NULL,
                exit_altitude_msl_m REAL NOT NULL,
                exit_altitude_agl_m REAL,
                ground_elevation_m REAL,
                is_valid_altitude INTEGER NOT NULL,
                sample_rate_hz REAL NOT NULL,
                quality_score REAL NOT NULL,
                quality_flags TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                jump_id TEXT NOT NULL,
                time_utc TEXT NOT NULL,
                t_rel_s REAL NOT NULL,
                lat REAL,
                lon REAL,
                hMSL_m REAL NOT NULL,
                hAGL_m REAL,
                velN_mps REAL NOT NULL,
                velE_mps REAL NOT NULL,
                velD_mps REAL NOT NULL,
                vVert_kmh REAL NOT NULL,
                vHor_kmh REAL NOT NULL,
                vTotal_kmh REAL NOT NULL,
                angle_deg REAL NOT NULL,
                accVert_mps2 REAL,
                hAcc REAL,
                vAcc REAL,
                sAcc REAL,
                gpsFix INTEGER,
                numSV INTEGER,
                quality_flags TEXT NOT NULL,
                FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_samples_jump_id ON samples(jump_id);
            CREATE INDEX IF NOT EXISTS idx_jumps_jumper_name ON jumps(jumper_name);

            CREATE TABLE IF NOT EXISTS metrics (
                jump_id TEXT PRIMARY KEY,
                best_3s_start_s REAL NOT NULL,
                best_3s_end_s REAL NOT NULL,
                best_3s_vVert_mps REAL NOT NULL,
                best_3s_vVert_kmh REAL NOT NULL,
                best_3s_vHor_kmh REAL,
                best_3s_angle_deg REAL,
                training_3s_max_from_t0 REAL NOT NULL,
                rule_based_3s_score REAL,
                rule_based_3s_score_mps REAL,
                performance_window_start_s REAL,
                performance_window_end_s REAL,
                validation_window_quality REAL,
                hot_zone_start_s REAL,
                hot_zone_end_s REAL,
                negative_risk_score REAL NOT NULL,
                notes TEXT,
                fixpoints_json TEXT NOT NULL,
                phases_json TEXT NOT NULL,
                scorecard_json TEXT NOT NULL,
                tips_json TEXT NOT NULL,
                quality_flags TEXT NOT NULL,
                FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()

