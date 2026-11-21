"""MySQL access (timelineofwealth): prices, index levels, Focus Five holdings and TWRR returns."""
import pandas as pd
from cryptography.fernet import Fernet
from dotenv import dotenv_values
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import URL


def get_engine(env_file):
    cfg = dotenv_values(env_file)
    needed = ("DB_HOST", "DB_USER", "DB_NAME", "DB_PASSWORD_ENC", "ENCRYPTION_KEY")
    missing = [k for k in needed if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"{env_file} is missing {missing}")
    password = Fernet(cfg["ENCRYPTION_KEY"].encode()).decrypt(cfg["DB_PASSWORD_ENC"].encode()).decode()
    url = URL.create("mysql+pymysql", username=cfg["DB_USER"], password=password,
                     host=cfg["DB_HOST"], database=cfg["DB_NAME"])
    return create_engine(url)


def read_sql(engine, sql, **params):
    stmt = text(sql)
    for key, value in params.items():
        if isinstance(value, (list, tuple, set)):
            stmt = stmt.bindparams(bindparam(key, expanding=True))
    clean = {k: list(v) if isinstance(v, (set, tuple)) else v for k, v in params.items()}
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn, params=clean)


FOCUS_HOLDINGS_SQL = """
SELECT b.description, a.memberid, a.portfolioid, a.ticker, a.short_name,
       SUM(a.quantity) AS quantity, SUM(a.total_cost) AS total_cost,
       SUM(a.market_value) AS market_value, SUM(a.net_profit) AS net_profit,
       SUM(a.market_value) / AVG(b.market_value) AS weight,
       AVG(b.market_value) AS portfolio_value, MIN(a.buy_date) AS first_buy
FROM portfolio_holdings a, portfolio b
WHERE a.portfolioid = b.portfolioid AND a.memberid = b.memberid
  AND a.memberid IN (:memberid) AND b.compositeid IN (:compositeid)
GROUP BY a.memberid, a.portfolioid, a.ticker
ORDER BY a.memberid, a.portfolioid, SUM(a.market_value) DESC
"""

PORTFOLIO_TWRR_SQL = """
SELECT returns_date,
       returns_twrr_since_current_month AS MTD, returns_twrr_three_months AS `3M`,
       returns_twrr_half_year AS `6M`, returns_twrr_one_year AS `1Y`, returns_twrr_ytd AS YTD,
       returns_twrr_two_year AS `2Y`, returns_twrr_three_year AS `3Y`, returns_twrr_five_year AS `5Y`
FROM portfolio_twrr_summary a
WHERE a.memberid = :memberid AND a.portfolioid = :portfolioid
"""

INDEX_TWRR_SQL = """
SELECT b.benchmark_name AS name, b.benchmark_type AS type, b.date_last_returns_process AS as_of,
       a.returns_twrr_since_current_month AS MTD, a.returns_twrr_three_months AS `3M`,
       a.returns_twrr_half_year AS `6M`, a.returns_twrr_one_year AS `1Y`, a.returns_twrr_ytd AS YTD,
       a.returns_twrr_two_year AS `2Y`, a.returns_twrr_three_year AS `3Y`, a.returns_twrr_five_year AS `5Y`
FROM benchmark_twrr_summary a, benchmark b
WHERE a.benchmarkid = b.benchmarkid AND b.is_mutual_fund = 'No'
ORDER BY b.benchmark_type, a.benchmarkid
"""


def load_focus_holdings(engine, memberid, compositeid):
    return read_sql(engine, FOCUS_HOLDINGS_SQL, memberid=memberid, compositeid=compositeid)


def load_portfolio_twrr(engine, memberid, portfolioid):
    return read_sql(engine, PORTFOLIO_TWRR_SQL, memberid=memberid, portfolioid=portfolioid)


def load_index_twrr(engine):
    return read_sql(engine, INDEX_TWRR_SQL)


def load_prices(engine, tickers, start):
    sql = """SELECT nse_ticker AS ticker, date, close_price AS close
             FROM nse_price_history WHERE nse_ticker IN :tickers AND date >= :start"""
    df = read_sql(engine, sql, tickers=sorted(tickers), start=start)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df["ticker"] = df["ticker"].str.upper()
    return df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()


def load_index_levels(engine, names, start):
    sql = "SELECT ticker, date, value FROM index_valuation WHERE ticker IN :names AND date >= :start"
    df = read_sql(engine, sql, names=list(names), start=start)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].astype(float)
    return df.pivot_table(index="date", columns="ticker", values="value", aggfunc="last").sort_index()


def load_stock_meta(engine, tickers):
    sql = """SELECT ticker, short_name, name, is_fno, marketcap
             FROM stock_universe WHERE ticker IN :tickers"""
    df = read_sql(engine, sql, tickers=sorted(tickers))
    df["ticker"] = df["ticker"].str.upper()
    df["marketcap"] = df["marketcap"].astype(float)
    return df.drop_duplicates("ticker").set_index("ticker")


def load_split_events(engine, tickers, start):
    sql = """SELECT ticker, date, close_price, previous_close_price, day_percent_change, is_processed, note
             FROM stock_split_probability WHERE ticker IN :tickers AND date >= :start ORDER BY date"""
    df = read_sql(engine, sql, tickers=sorted(tickers), start=start)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].str.upper()
    for col in ("close_price", "previous_close_price", "day_percent_change"):
        df[col] = df[col].astype(float)
    return df


def latest_price_date(engine):
    return pd.to_datetime(read_sql(engine, "SELECT MAX(date) AS d FROM nse_price_history")["d"].iloc[0])


def load_ticker_renames(engine):
    """Old NSE symbol -> current symbol, from stock_universe.ticker_old (e.g. ICICIPRULI -> ICICILIFE).
    Numeric ticker_old values are BSE codes, not renames, so they are skipped."""
    df = read_sql(engine, """SELECT ticker, ticker_old FROM stock_universe
                             WHERE ticker_old IS NOT NULL AND ticker_old <> '' AND ticker_old <> ticker""")
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["ticker_old"] = df["ticker_old"].str.strip().str.upper()
    df = df[~df["ticker_old"].str.fullmatch(r"\d+")]
    return dict(zip(df["ticker_old"], df["ticker"]))


def load_known_tickers(engine):
    """Every symbol in stock_universe (used to validate tickers taken from report file names)."""
    df = read_sql(engine, "SELECT ticker FROM stock_universe")
    return set(df["ticker"].str.strip().str.upper())


def load_bse_symbols(engine):
    """BSE-only listings are filed under their BSE code (FY27Q2_544937_MOSL.pdf) but tracked on Yahoo by a symbol
    (NSE); stock_universe keeps that symbol in short_name. Returns {bse code: symbol} where short_name is one word."""
    df = read_sql(engine, "SELECT ticker, short_name FROM stock_universe WHERE ticker REGEXP '^[0-9]+$'")
    df["short_name"] = df["short_name"].fillna("").str.strip()
    df = df[df["short_name"].str.fullmatch(r"[A-Z0-9&\-]+")]
    return dict(zip(df["ticker"].str.strip(), df["short_name"]))
