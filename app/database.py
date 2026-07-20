import sqlite3
from contextlib import contextmanager

from app.config import DATABASE_PATH


def _migration_001_performance_storage(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jump_series (
            jump_id TEXT PRIMARY KEY,
            series_version INTEGER NOT NULL DEFAULT 1,
            encoding TEXT NOT NULL,
            point_count INTEGER NOT NULL,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            payload BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS jump_analysis_features (
            jump_id TEXT PRIMARY KEY,
            feature_version INTEGER NOT NULL DEFAULT 1,
            reference_signature TEXT NOT NULL DEFAULT '',
            source_json TEXT NOT NULL,
            record_json TEXT,
            reference_eligible INTEGER NOT NULL DEFAULT 0 CHECK(reference_eligible IN (0, 1)),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_jump_analysis_features_reference
        ON jump_analysis_features(reference_eligible, jump_id);

        CREATE TABLE IF NOT EXISTS jumper_profile_snapshots (
            jumper_key TEXT PRIMARY KEY,
            jumper_name TEXT NOT NULL,
            feature_version INTEGER NOT NULL DEFAULT 1,
            history_signature TEXT NOT NULL,
            reference_signature TEXT NOT NULL DEFAULT '',
            jump_count INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ai_coaching_results (
            cache_key TEXT PRIMARY KEY,
            jump_id TEXT,
            analysis_signature TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            view_mode TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('ready', 'error')),
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_ai_coaching_results_jump
        ON ai_coaching_results(jump_id, updated_at DESC);
        """
    )


_SCHEMA_MIGRATIONS = (
    (1, "performance_storage_v1", _migration_001_performance_storage),
)


def _apply_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    applied = {
        int(row["version"])
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for version, name, migration in _SCHEMA_MIGRATIONS:
        if version in applied:
            continue
        migration(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
            (version, name),
        )


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA temp_store = MEMORY;")
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
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA wal_autocheckpoint = 1000;")
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS jumps (
                jump_id TEXT PRIMARY KEY,
                jumper_name TEXT NOT NULL,
                file_name TEXT NOT NULL,
                device_type TEXT NOT NULL,
                is_reference_only INTEGER NOT NULL DEFAULT 0,
                jump_context TEXT NOT NULL DEFAULT 'unknown',
                source_file_sha256 TEXT,
                source_file_path TEXT,
                raw_start_time_utc TEXT NOT NULL,
                t0_utc TEXT NOT NULL,
                exit_altitude_msl_m REAL NOT NULL,
                exit_altitude_agl_m REAL,
                ground_elevation_m REAL,
                ground_elevation_source TEXT NOT NULL DEFAULT 'estimated',
                breakoff_altitude_agl_m REAL NOT NULL DEFAULT 1707.0,
                analysis_version TEXT NOT NULL DEFAULT 'legacy',
                analysis_signature TEXT NOT NULL DEFAULT 'legacy',
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
            CREATE INDEX IF NOT EXISTS idx_samples_jump_id_t_rel_s ON samples(jump_id, t_rel_s);
            CREATE INDEX IF NOT EXISTS idx_jumps_jumper_name ON jumps(jumper_name);
            CREATE INDEX IF NOT EXISTS idx_jumps_active_recent ON jumps(is_reference_only, t0_utc DESC, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_jumps_jumper_active_recent ON jumps(jumper_name, is_reference_only, t0_utc DESC, created_at DESC);

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
                validation_window_start_s REAL,
                validation_window_end_s REAL,
                rule_score_status TEXT NOT NULL DEFAULT 'estimated',
                rule_score_reasons TEXT NOT NULL DEFAULT '[]',
                analysis_version TEXT NOT NULL DEFAULT 'legacy',
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

            CREATE INDEX IF NOT EXISTS idx_metrics_best_3s ON metrics(best_3s_vVert_kmh DESC);

            CREATE TABLE IF NOT EXISTS jump_feedback (
                jump_id TEXT PRIMARY KEY,
                feedback_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS coaching_snapshots (
                jump_id TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS dropzones (
                dropzone_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                country_code TEXT NOT NULL,
                region TEXT,
                locality TEXT,
                icao_code TEXT,
                airport_ident TEXT,
                latitude REAL NOT NULL CHECK(latitude BETWEEN -90.0 AND 90.0),
                longitude REAL NOT NULL CHECK(longitude BETWEEN -180.0 AND 180.0),
                match_radius_m REAL NOT NULL DEFAULT 1500.0 CHECK(match_radius_m > 0.0),
                ground_elevation_m REAL,
                published_elevation_m REAL,
                ground_elevation_uncertainty_m REAL,
                ground_elevation_source TEXT,
                status TEXT NOT NULL DEFAULT 'candidate'
                    CHECK(status IN ('candidate', 'trusted', 'verified', 'inactive')),
                catalog_revision INTEGER NOT NULL DEFAULT 1 CHECK(catalog_revision >= 1),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_dropzones_country_airport_ident
            ON dropzones(country_code, airport_ident)
            WHERE airport_ident IS NOT NULL AND airport_ident <> '';
            CREATE INDEX IF NOT EXISTS idx_dropzones_status_country
            ON dropzones(status, country_code);
            CREATE INDEX IF NOT EXISTS idx_dropzones_coordinates
            ON dropzones(latitude, longitude);

            CREATE TABLE IF NOT EXISTS dropzone_zones (
                zone_id TEXT PRIMARY KEY,
                dropzone_id TEXT NOT NULL,
                name TEXT NOT NULL,
                zone_kind TEXT NOT NULL DEFAULT 'landing'
                    CHECK(zone_kind IN ('primary', 'landing', 'alternate', 'historical')),
                latitude REAL NOT NULL CHECK(latitude BETWEEN -90.0 AND 90.0),
                longitude REAL NOT NULL CHECK(longitude BETWEEN -180.0 AND 180.0),
                match_radius_m REAL NOT NULL DEFAULT 1500.0 CHECK(match_radius_m > 0.0),
                ground_elevation_m REAL,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'candidate', 'inactive')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(dropzone_id) REFERENCES dropzones(dropzone_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_dropzone_zones_dropzone
            ON dropzone_zones(dropzone_id);
            CREATE INDEX IF NOT EXISTS idx_dropzone_zones_coordinates
            ON dropzone_zones(latitude, longitude);

            CREATE TABLE IF NOT EXISTS dropzone_operators (
                operator_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                country_code TEXT NOT NULL,
                website_url TEXT,
                status TEXT NOT NULL DEFAULT 'listed'
                    CHECK(status IN ('listed', 'confirmed', 'inactive')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS dropzone_operator_assignments (
                dropzone_id TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                is_primary INTEGER NOT NULL DEFAULT 0 CHECK(is_primary IN (0, 1)),
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'candidate', 'inactive')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(dropzone_id, operator_id),
                FOREIGN KEY(dropzone_id) REFERENCES dropzones(dropzone_id) ON DELETE CASCADE,
                FOREIGN KEY(operator_id) REFERENCES dropzone_operators(operator_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_dropzone_operator_assignments_operator
            ON dropzone_operator_assignments(operator_id);

            CREATE TABLE IF NOT EXISTS dropzone_sources (
                source_id TEXT PRIMARY KEY,
                dropzone_id TEXT,
                zone_id TEXT,
                operator_id TEXT,
                source_kind TEXT NOT NULL
                    CHECK(source_kind IN (
                        'directory', 'official', 'airport_registry', 'terrain',
                        'historical_gps', 'manual'
                    )),
                source_name TEXT NOT NULL,
                source_url TEXT,
                source_ref TEXT,
                trust_level TEXT NOT NULL DEFAULT 'supporting'
                    CHECK(trust_level IN ('discovery', 'supporting', 'primary')),
                supports_fields_json TEXT NOT NULL DEFAULT '[]',
                details_json TEXT NOT NULL DEFAULT '{}',
                retrieved_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(dropzone_id) REFERENCES dropzones(dropzone_id) ON DELETE CASCADE,
                FOREIGN KEY(zone_id) REFERENCES dropzone_zones(zone_id) ON DELETE CASCADE,
                FOREIGN KEY(operator_id) REFERENCES dropzone_operators(operator_id) ON DELETE CASCADE,
                CHECK(dropzone_id IS NOT NULL OR zone_id IS NOT NULL OR operator_id IS NOT NULL)
            );

            CREATE INDEX IF NOT EXISTS idx_dropzone_sources_dropzone
            ON dropzone_sources(dropzone_id);
            CREATE INDEX IF NOT EXISTS idx_dropzone_sources_operator
            ON dropzone_sources(operator_id);
            CREATE INDEX IF NOT EXISTS idx_dropzone_sources_kind
            ON dropzone_sources(source_kind);

            CREATE TABLE IF NOT EXISTS dropzone_observations (
                observation_id TEXT PRIMARY KEY,
                dropzone_id TEXT NOT NULL,
                jump_id TEXT,
                observed_at TEXT,
                latitude REAL NOT NULL CHECK(latitude BETWEEN -90.0 AND 90.0),
                longitude REAL NOT NULL CHECK(longitude BETWEEN -180.0 AND 180.0),
                ground_elevation_m REAL,
                altitude_mad_m REAL,
                sample_count INTEGER,
                duration_s REAL,
                source_kind TEXT NOT NULL DEFAULT 'historical_gps',
                quality_status TEXT NOT NULL DEFAULT 'candidate'
                    CHECK(quality_status IN ('candidate', 'accepted', 'rejected')),
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(dropzone_id) REFERENCES dropzones(dropzone_id) ON DELETE CASCADE,
                FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_dropzone_observations_dropzone
            ON dropzone_observations(dropzone_id);
            CREATE INDEX IF NOT EXISTS idx_dropzone_observations_jump
            ON dropzone_observations(jump_id);

            CREATE TABLE IF NOT EXISTS dropzone_match_attempts (
                attempt_id TEXT PRIMARY KEY,
                jump_id TEXT NOT NULL,
                algorithm_version TEXT NOT NULL,
                catalog_version TEXT,
                match_status TEXT NOT NULL
                    CHECK(match_status IN (
                        'accepted', 'manual', 'candidate', 'ambiguous', 'unmatched',
                        'low_confidence', 'insufficient', 'unavailable'
                    )),
                assignment_source TEXT,
                observed_at TEXT,
                latitude REAL CHECK(latitude IS NULL OR latitude BETWEEN -90.0 AND 90.0),
                longitude REAL CHECK(longitude IS NULL OR longitude BETWEEN -180.0 AND 180.0),
                ground_elevation_m REAL,
                altitude_mad_m REAL,
                coordinate_p95_radius_m REAL,
                sample_count INTEGER,
                duration_s REAL,
                nearest_dropzone_id TEXT,
                nearest_zone_id TEXT,
                nearest_distance_m REAL,
                second_distance_m REAL,
                confidence REAL CHECK(confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0),
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(jump_id) REFERENCES jumps(jump_id) ON DELETE CASCADE,
                FOREIGN KEY(nearest_dropzone_id) REFERENCES dropzones(dropzone_id) ON DELETE SET NULL,
                FOREIGN KEY(nearest_zone_id) REFERENCES dropzone_zones(zone_id) ON DELETE SET NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_dropzone_match_attempts_jump_algorithm
            ON dropzone_match_attempts(jump_id, algorithm_version);
            CREATE INDEX IF NOT EXISTS idx_dropzone_match_attempts_status
            ON dropzone_match_attempts(match_status, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_dropzone_match_attempts_coordinates
            ON dropzone_match_attempts(latitude, longitude);

            CREATE TABLE IF NOT EXISTS dropzone_catalog_metadata (
                catalog_key TEXT PRIMARY KEY,
                catalog_version TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                dropzone_count INTEGER NOT NULL,
                operator_count INTEGER NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                audit_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )

        # Lightweight migration path for existing DB files.
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(jumps)").fetchall()
        }
        if "source_file_sha256" not in columns:
            conn.execute("ALTER TABLE jumps ADD COLUMN source_file_sha256 TEXT")
        if "source_file_path" not in columns:
            conn.execute("ALTER TABLE jumps ADD COLUMN source_file_path TEXT")
        if "is_reference_only" not in columns:
            conn.execute(
                "ALTER TABLE jumps ADD COLUMN is_reference_only INTEGER NOT NULL DEFAULT 0"
            )
        if "jump_context" not in columns:
            conn.execute(
                "ALTER TABLE jumps ADD COLUMN jump_context TEXT NOT NULL DEFAULT 'unknown'"
            )
        if "ground_elevation_source" not in columns:
            conn.execute(
                "ALTER TABLE jumps ADD COLUMN ground_elevation_source TEXT NOT NULL DEFAULT 'estimated'"
            )
        if "breakoff_altitude_agl_m" not in columns:
            conn.execute(
                "ALTER TABLE jumps ADD COLUMN breakoff_altitude_agl_m REAL NOT NULL DEFAULT 1707.0"
            )
        if "analysis_version" not in columns:
            conn.execute(
                "ALTER TABLE jumps ADD COLUMN analysis_version TEXT NOT NULL DEFAULT 'legacy'"
            )
        if "analysis_signature" not in columns:
            conn.execute(
                "ALTER TABLE jumps ADD COLUMN analysis_signature TEXT NOT NULL DEFAULT 'legacy'"
            )
        if "dropzone_id" not in columns:
            conn.execute(
                "ALTER TABLE jumps ADD COLUMN dropzone_id TEXT REFERENCES dropzones(dropzone_id) ON DELETE SET NULL"
            )
        if "dropzone_zone_id" not in columns:
            conn.execute(
                "ALTER TABLE jumps ADD COLUMN dropzone_zone_id TEXT REFERENCES dropzone_zones(zone_id) ON DELETE SET NULL"
            )
        if "dropzone_revision" not in columns:
            conn.execute("ALTER TABLE jumps ADD COLUMN dropzone_revision INTEGER")
        if "dropzone_assignment_source" not in columns:
            conn.execute("ALTER TABLE jumps ADD COLUMN dropzone_assignment_source TEXT")
        if "dropzone_assignment_confidence" not in columns:
            conn.execute("ALTER TABLE jumps ADD COLUMN dropzone_assignment_confidence REAL")
        if "dropzone_distance_m" not in columns:
            conn.execute("ALTER TABLE jumps ADD COLUMN dropzone_distance_m REAL")
        conn.execute(
            """
            UPDATE jumps
            SET ground_elevation_source = CASE
                WHEN quality_flags LIKE '%NO_GROUND_LEVEL%' THEN 'estimated'
                ELSE 'manual'
            END
            WHERE analysis_version = 'legacy'
            """
        )
        metric_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(metrics)").fetchall()
        }
        if "validation_window_start_s" not in metric_columns:
            conn.execute("ALTER TABLE metrics ADD COLUMN validation_window_start_s REAL")
        if "validation_window_end_s" not in metric_columns:
            conn.execute("ALTER TABLE metrics ADD COLUMN validation_window_end_s REAL")
        if "rule_score_status" not in metric_columns:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN rule_score_status TEXT NOT NULL DEFAULT 'estimated'"
            )
        if "rule_score_reasons" not in metric_columns:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN rule_score_reasons TEXT NOT NULL DEFAULT '[]'"
            )
        if "analysis_version" not in metric_columns:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN analysis_version TEXT NOT NULL DEFAULT 'legacy'"
            )
        conn.execute(
            "UPDATE metrics SET rule_score_status = 'estimated' WHERE analysis_version = 'legacy'"
        )
        conn.execute("DROP INDEX IF EXISTS idx_jumps_source_file_sha256_unique")
        conn.execute("DROP INDEX IF EXISTS idx_jumps_jumper_hash_unique")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_jumps_jumper_hash_signature_unique "
            "ON jumps(jumper_name, source_file_sha256, analysis_signature)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jumps_dropzone_id ON jumps(dropzone_id)"
        )
        _apply_schema_migrations(conn)
        conn.commit()
