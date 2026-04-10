"""SQLite schema initialisation -- creates tables and inserts default settings."""

import aiosqlite
import config as cfg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    slot_start TEXT NOT NULL,
    slot_end TEXT NOT NULL,
    slot_timestamp INTEGER NOT NULL,
    side TEXT,
    entry_price REAL,
    opposite_price REAL,
    outcome TEXT,
    is_win INTEGER,
    resolved_at TIMESTAMP,
    skipped INTEGER DEFAULT 0,
    filter_blocked INTEGER DEFAULT 0,
    pattern TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    slot_start TEXT NOT NULL,
    slot_end TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    amount_usdc REAL NOT NULL,
    order_id TEXT,
    fill_price REAL,
    status TEXT DEFAULT 'pending',
    outcome TEXT,
    is_win INTEGER,
    pnl REAL,
    resolved_at TIMESTAMP,
    retry_count INTEGER DEFAULT 0,
    last_retry_at TIMESTAMP,
    is_demo INTEGER DEFAULT 0,
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS redemptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    condition_id TEXT NOT NULL,
    outcome_index INTEGER NOT NULL,
    size REAL NOT NULL,
    title TEXT,
    tx_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    gas_used INTEGER,
    dry_run INTEGER NOT NULL DEFAULT 0,
    resolved_at TIMESTAMP,
    verified INTEGER NOT NULL DEFAULT 0,
    verified_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    slot TEXT NOT NULL,
    train_date TEXT,
    wr REAL,
    precision_score REAL,
    trades_per_day REAL,
    threshold REAL,
    sample_count INTEGER,
    path TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS model_blobs (
    slot TEXT PRIMARY KEY,
    blob BLOB NOT NULL,
    metadata TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

DEFAULT_SETTINGS = {
    "autotrade_enabled": "false",
    "trade_amount_usdc": str(cfg.TRADE_AMOUNT_USDC),
    "trade_mode": cfg.TRADE_MODE,
    "trade_pct": str(cfg.TRADE_PCT),
    "auto_redeem_enabled": "false",
    "demo_trade_enabled": "false",
    "demo_bankroll_usdc": "1000.00",
    "invert_trades_enabled": "false",
}


async def init_db(db_path: str | None = None) -> None:
    """Create tables if they don't exist and seed default settings."""
    path = db_path or cfg.DB_PATH
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA_SQL)
        for key, value in DEFAULT_SETTINGS.items():
            await db.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        # Seed default ML thresholds (INSERT OR IGNORE — never overwrite live values)
        await db.execute(
            "INSERT OR IGNORE INTO ml_config (key, value) VALUES ('ml_threshold', '0.53')"
        )
        await db.execute(
            "INSERT OR IGNORE INTO ml_config (key, value) VALUES ('ml_down_threshold', '0.47')"
        )
        await db.commit()


# The 4 condition IDs incorrectly redeemed from the wrong address (sig-type-2 bug).
# Deleting these records lets the redeemer retry them on the next scan.
_BAD_CONDITION_IDS = [
    "0x46b556649c109de10c5be1be2dbc4ee3155909fee0d99230e17dbd51020fcb35",
    "0x1b447392bdf148658a553757511a4a9320ec36486ac42727fbe7c93a192158ae",
    "0x0fe4e91b6df78899d791e19fdf8176d8bcf242fde888190115fa66dc4b724d85",
    "0x6daf71ed6a57d96e62563df405159ef67ccfcdd1206e8139ef417c03ba4b26c7",
]


async def cleanup_bad_redemptions(db_path: str | None = None) -> int:
    """One-time startup cleanup: delete incorrectly recorded redemption rows.

    These 4 conditions were broadcast from the wrong address (EOA instead of
    the proxy wallet) due to the sig-type-2 bug.  Removing the 'success'
    records allows the redeemer to retry them on the next scan.

    Safe to run repeatedly -- if no rows match, rowcount is 0.
    Returns total rows deleted.
    """
    path = db_path or cfg.DB_PATH
    total = 0
    async with aiosqlite.connect(path) as db:
        for cid in _BAD_CONDITION_IDS:
            cursor = await db.execute(
                "DELETE FROM redemptions WHERE condition_id = ? AND dry_run = 0",
                (cid,),
            )
            total += cursor.rowcount
        await db.commit()
    return total


async def migrate_db(db_path: str | None = None) -> None:
    """Add new columns/tables if they don't exist (safe to run repeatedly).

    Every DDL step is wrapped in its own try/except so a single failure
    never aborts the rest of the migration.
    """
    import logging
    log = logging.getLogger(__name__)
    path = db_path or cfg.DB_PATH

    async with aiosqlite.connect(path) as db:

        # --- trades columns ---
        try:
            cursor = await db.execute("PRAGMA table_info(trades)")
            columns = {row[1] for row in await cursor.fetchall()}
            if "retry_count" not in columns:
                await db.execute("ALTER TABLE trades ADD COLUMN retry_count INTEGER DEFAULT 0")
            if "last_retry_at" not in columns:
                await db.execute("ALTER TABLE trades ADD COLUMN last_retry_at TIMESTAMP")
            if "is_demo" not in columns:
                await db.execute("ALTER TABLE trades ADD COLUMN is_demo INTEGER DEFAULT 0")
        except Exception as e:
            log.warning("migrate_db: trades column migration failed: %s", e)

        # --- signals columns ---
        try:
            cursor2 = await db.execute("PRAGMA table_info(signals)")
            sig_columns = {row[1] for row in await cursor2.fetchall()}
            if "filter_blocked" not in sig_columns:
                await db.execute("ALTER TABLE signals ADD COLUMN filter_blocked INTEGER DEFAULT 0")
            if "pattern" not in sig_columns:
                await db.execute("ALTER TABLE signals ADD COLUMN pattern TEXT")
        except Exception as e:
            log.warning("migrate_db: signals column migration failed: %s", e)

        # --- redemptions columns ---
        try:
            cursor3 = await db.execute("PRAGMA table_info(redemptions)")
            red_columns = {row[1] for row in await cursor3.fetchall()}
            if "verified" not in red_columns:
                await db.execute(
                    "ALTER TABLE redemptions ADD COLUMN verified INTEGER NOT NULL DEFAULT 0"
                )
            if "verified_at" not in red_columns:
                await db.execute(
                    "ALTER TABLE redemptions ADD COLUMN verified_at TIMESTAMP"
                )
        except Exception as e:
            log.warning("migrate_db: redemptions column migration failed: %s", e)

        # --- ml_config table ---
        try:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS ml_config (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
        except Exception as e:
            log.warning("migrate_db: ml_config table creation failed: %s", e)

        # --- model_registry table ---
        try:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    slot TEXT NOT NULL,
                    train_date TEXT,
                    wr REAL,
                    precision_score REAL,
                    trades_per_day REAL,
                    threshold REAL,
                    sample_count INTEGER,
                    path TEXT,
                    metadata TEXT
                )
            """)
        except Exception as e:
            log.warning("migrate_db: model_registry table creation failed: %s", e)

        # --- model_blobs table (DB-persisted model storage) ---
        try:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_blobs (
                    slot TEXT PRIMARY KEY,
                    blob BLOB NOT NULL,
                    metadata TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        except Exception as e:
            log.warning("migrate_db: model_blobs table creation failed: %s", e)

        # --- seed default ML threshold ---
        try:
            await db.execute(
                "INSERT OR IGNORE INTO ml_config (key, value) VALUES ('ml_threshold', '0.56')"
            )
        except Exception as e:
            log.warning("migrate_db: ml_threshold seed failed: %s", e)

        # --- seed default ML DOWN threshold (1 - up_threshold = 1 - 0.56 = 0.44) ---
        try:
            await db.execute(
                "INSERT OR IGNORE INTO ml_config (key, value) VALUES ('ml_down_threshold', '0.44')"
            )
        except Exception as e:
            log.warning("migrate_db: ml_down_threshold seed failed: %s", e)

        # --- seed default settings ---
        for key, value in DEFAULT_SETTINGS.items():
            try:
                await db.execute(
                    "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                    (key, value),
                )
            except Exception as e:
                log.warning("migrate_db: settings seed failed for key=%s: %s", key, e)

        await db.commit()
