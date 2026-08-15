"""Project-wide constants and canonical column names.

Centralising these avoids magic strings scattered across modules and makes the
canonical market-data schema a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Filesystem layout. `pathlib` keeps path construction portable across
# Windows, macOS and Linux. A development checkout stores artefacts under the
# repository root; an installed package uses the per-user `~/.quantlab` home.
# --------------------------------------------------------------------------- #
#: Directory containing the installed or source-checkout ``quantlab`` package.
_PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent


def _detect_project_root() -> Path:
    """Locate the development repository root or use ``~/.quantlab``.

    A regular wheel has no repository-level ``src/`` layout, so installed-package
    data, reports and logs must not be derived from the package's location inside
    ``site-packages``.
    """
    candidate = _PACKAGE_DIR.parents[1]
    if (candidate / "pyproject.toml").is_file() and (
        candidate / "src" / "quantlab"
    ).is_dir():
        return candidate
    return Path.home() / ".quantlab"


PROJECT_ROOT: Final[Path] = _detect_project_root()
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
METADATA_DIR: Final[Path] = DATA_DIR / "metadata"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
GENERATED_REPORTS_DIR: Final[Path] = REPORTS_DIR / "generated"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
#: Shipped experiment configurations. Wheels bundle them under the package;
#: development checkouts use the repository-level ``configs/`` directory.
CONFIGS_DIR: Final[Path] = (
    _PACKAGE_DIR / "configs"
    if (_PACKAGE_DIR / "configs").is_dir()
    else PROJECT_ROOT / "configs"
)
#: Synthetic CSV data bundled for ``demo_offline.yaml``. A user-provided file
#: under ``RAW_DATA_DIR`` takes priority over this packaged fallback.
DEMO_DATA_DIR: Final[Path] = _PACKAGE_DIR / "demo_data"

# --------------------------------------------------------------------------- #
# Canonical market-data schema. Every data source must be normalised to
# these columns before entering the rest of the pipeline.
# --------------------------------------------------------------------------- #
TIMESTAMP: Final[str] = "timestamp"
SYMBOL: Final[str] = "symbol"
OPEN: Final[str] = "open"
HIGH: Final[str] = "high"
LOW: Final[str] = "low"
CLOSE: Final[str] = "close"
ADJUSTED_CLOSE: Final[str] = "adjusted_close"
VOLUME: Final[str] = "volume"

#: Columns required in every normalised market-data frame, in canonical order.
OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    TIMESTAMP,
    SYMBOL,
    OPEN,
    HIGH,
    LOW,
    CLOSE,
    ADJUSTED_CLOSE,
    VOLUME,
)

#: Price columns that must be strictly positive.
PRICE_COLUMNS: Final[tuple[str, ...]] = (OPEN, HIGH, LOW, CLOSE, ADJUSTED_CLOSE)

# --------------------------------------------------------------------------- #
# Annualisation factors. Choosing the right factor is essential to
# avoid the classic "annualise incorrectly" mistake.
# --------------------------------------------------------------------------- #
TRADING_DAYS_PER_YEAR: Final[int] = 252
CALENDAR_DAYS_PER_YEAR: Final[int] = 365
WEEKS_PER_YEAR: Final[int] = 52
MONTHS_PER_YEAR: Final[int] = 12

#: Maps a pandas-style frequency string to a default number of periods/year,
#: for markets with a fixed trading calendar (equities/ETFs). Used when the
#: config does not pin ``periods_per_year`` explicitly.
FREQUENCY_TO_PERIODS_PER_YEAR: Final[dict[str, int]] = {
    "1d": TRADING_DAYS_PER_YEAR,
    "1D": TRADING_DAYS_PER_YEAR,
    "1h": TRADING_DAYS_PER_YEAR * 7,  # ~6.5h sessions rounded; overridable
    "1H": TRADING_DAYS_PER_YEAR * 7,
    "1w": WEEKS_PER_YEAR,
    "1W": WEEKS_PER_YEAR,
    "1mo": MONTHS_PER_YEAR,
    "1M": MONTHS_PER_YEAR,
}

#: Same, for 24/7 markets (crypto). A single global table for both would
#: either overstate a stock's trading hours or understate a crypto asset's —
#: e.g. hourly bars trade ~1,764 times/year on an equity calendar but 8,760
#: times/year around the clock, a ~5x difference that would silently distort
#: every annualised metric if the two were conflated.
CRYPTO_FREQUENCY_TO_PERIODS_PER_YEAR: Final[dict[str, int]] = {
    "1d": CALENDAR_DAYS_PER_YEAR,
    "1D": CALENDAR_DAYS_PER_YEAR,
    "1h": 24 * CALENDAR_DAYS_PER_YEAR,
    "1H": 24 * CALENDAR_DAYS_PER_YEAR,
    "1w": WEEKS_PER_YEAR,
    "1W": WEEKS_PER_YEAR,
    # Monthly bars always annualise to 12 periods, for both equity and 24/7 markets.
    "1mo": MONTHS_PER_YEAR,
    "1M": MONTHS_PER_YEAR,
}

# --------------------------------------------------------------------------- #
# Cost model conversion. Basis points → fraction.
# --------------------------------------------------------------------------- #
BPS_TO_FRACTION: Final[float] = 1.0 / 10_000.0

#: Small epsilon used to protect against divide-by-zero in volatility scaling,
#: inverse-volatility weights and volume-based slippage.
EPSILON: Final[float] = 1e-12

#: Default risk-free rate (annualised) when a config omits it.
DEFAULT_RISK_FREE_RATE: Final[float] = 0.0

# Package logger name — modules obtain children of this via logging_config.
LOGGER_NAME: Final[str] = "quantlab"
