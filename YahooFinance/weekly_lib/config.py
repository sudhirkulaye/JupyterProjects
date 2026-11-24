"""Settings shared by the two weekly programs: folders, portfolio ids and analysis parameters."""
from pathlib import Path

# ---------------------------------------------------------------- folders
STOCK_DIR = Path(r"C:\MyDocuments\03Business\05ResearchAndAnalysis\StockInvestments")
YC_DIR = STOCK_DIR / "ResearchReports" / "YahooConsensus"       # only the 4 .bat files + ticker list live here
TICKERS_FILE = YC_DIR / "YahooFinanceTickers.csv"
CONSENSUS_DIR = YC_DIR / "WeeklyConsensus"                       # yyyymmdd_YahooConsensus.csv files
HISTORY_FILE = CONSENSUS_DIR / "YahooConsensusHistory.csv"       # merged history (also uploaded to Cowork)
DASHBOARD_DIR = YC_DIR / "Dashboard"
LATEST_DASHBOARD = DASHBOARD_DIR / "WeeklyDashboard.html"
ARCHIVE_NAME = "{date}_WeeklyDashboard.html"
LOG_DIR = YC_DIR / "Logs"
OLD_DIR = YC_DIR / "Old"

REPORTS_DIR = STOCK_DIR / "ResearchReports" / "CompanyResearchReports"
ANALYST_TRACKER_XLSX = STOCK_DIR / "QuarterResultsScreenerExcels" / "DBInsert" / "AnalystViewTracker.xlsx"
ANALYST_RECO_SHEET = "AnalystsReco"
AMBIT_XLSX = STOCK_DIR / "CoworkProject" / "ResearchIntelligence" / "AnalystVeiwInsight.xlsx"
WATCHLISTS_FILE = Path(r"C:\MyDocuments\03Business\11Property\website\watchlists.properties")   # shared with the website
if not WATCHLISTS_FILE.exists():                                     # old location
    WATCHLISTS_FILE = STOCK_DIR / "QuarterResultsScreenerExcels" / "watchlists.properties"
DB_ENV_FILE = Path.home() / "JupyterProjects" / "DataBaseAccess" / "DBConfig.env"   # C:\Users\<user>\... on either laptop

# ---------------------------------------------------------------- download
NEW_COVERAGE_DAYS = 10            # report PDFs created / changed in the last N days count as new coverage
NEW_COVERAGE_BROKERS = ("MOSL", "AMBIT")
DOWNLOAD_PAUSE_SECONDS = 0.3      # gentle pacing between tickers
DOWNLOAD_RETRIES = 3

# ---------------------------------------------------------------- portfolio and watchlists
FOCUS_MEMBER_ID = 1
FOCUS_COMPOSITE_ID = 2
FOCUS_PORTFOLIO_ID = 2
CASH_TICKERS = {"MOSL_CASH"}
SHORTLIST_SECTIONS = [
    "002_Shortlisted",
    "003_ShortlistedLargeCap",
    "004_ShortlistedMidCap",
    "005_ShortlistedSmallCap",
]

# ---------------------------------------------------------------- benchmarks
MAIN_INDEX = "NIFTY"
INDICES = ["NIFTY", "NIFTY200", "BSEMidCap", "BSESmallCap"]

# ---------------------------------------------------------------- analysis parameters
MIN_ANALYSTS = 5             # consensus coverage needed to be scored
SCORE_WEEKS = 4              # look-back of the composite score components
FORWARD_WEEKS = 4            # holding period used by the scoreboard
TOP_N = 15                   # rows in the long / short idea lists
RETURN_WINDOWS = ["1W", "2W", "1M", "3M", "6M", "12M", "YTD"]
PRICE_HISTORY_START = "2025-10-01"   # index levels from here; stock prices start earlier when 12-1 momentum needs it
BROKERS = ["MOSL", "AMBIT", "PL", "AXIS"]
RECO_SCORE = {"BUY": 2, "ADD": 1, "ACCUMULATE": 1, "HOLD": 0, "NEUTRAL": 0, "REDUCE": -1, "SELL": -2}
BROKER_VIEW_MAX_AGE_DAYS = 270   # ignore a broker's last view if older than this
