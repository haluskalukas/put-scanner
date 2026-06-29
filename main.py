from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
import math
import re
import statistics
from datetime import datetime, timezone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120",
    "Accept": "application/json",
}

# ── Black-Scholes ──────────────────────────────────────────────────────────────

def erf(x):
    s = -1 if x < 0 else 1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*math.exp(-x*x)
    return s * y

def N(x):
    return 0.5 * (1 + erf(x / math.sqrt(2)))

def bs_put(S, K, T, r, sig):
    if T <= 0 or sig <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5*sig*sig)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    return K*math.exp(-r*T)*N(-d2) - S*N(-d1)

def bs_greeks(S, K, T, r, sig):
    if T <= 0 or sig <= 0:
        return {}
    d1 = (math.log(S / K) + (r + 0.5*sig*sig)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    delta = -N(-d1)
    gamma = math.exp(-d1*d1/2) / (S * sig * math.sqrt(T) * math.sqrt(2*math.pi))
    vega  = S * math.exp(-d1*d1/2) / math.sqrt(2*math.pi) * math.sqrt(T) / 100
    theta = (-(S * math.exp(-d1*d1/2) * sig) / (2 * math.sqrt(T) * math.sqrt(2*math.pi))
             + r * K * math.exp(-r*T) * N(-d2)) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

def solve_iv(price, S, K, T, r):
    if T <= 0 or price <= 0:
        return None
    lo, hi = 0.001, 5.0
    if price < bs_put(S, K, T, r, lo):
        return None
    for _ in range(80):
        m = (lo + hi) / 2
        p = bs_put(S, K, T, r, m)
        if abs(p - price) < 1e-5:
            return m
        if p > price:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2

def analyze_put(S, K, dte, premium, iv_raw, r, realized, rich_thresh,
                open_interest=0, volume=0, bid=0, ask=0, rv_rank=None):
    T = dte / 365
    sigma = iv_raw
    if (sigma is None or sigma <= 0) and premium > 0:
        sigma = solve_iv(premium, S, K, T, r)
    if sigma is None or sigma <= 0:
        return None

    greeks = bs_greeks(S, K, T, r, sigma)
    delta    = greeks.get("delta", 0)
    prob_itm = N(-(math.log(S/K) + (r - 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))) if T > 0 else (1 if K > S else 0)
    fair     = bs_put(S, K, T, r, sigma)
    breakeven = K - premium
    pop = None
    if breakeven > 0 and T > 0:
        dbe = (math.log(S / breakeven) + (r - 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
        pop = N(dbe)
    roc_annual = (premium / K) * (365 / dte) if dte > 0 else 0

    iv_rv = sigma / realized if realized > 0 else None

    # bid/ask spread quality: 0=ok, 1=wide, 2=unusable
    spread = ask - bid
    spread_pct = spread / premium if premium > 0 else 99
    spread_flag = "ok" if spread_pct < 0.15 else ("wide" if spread_pct < 0.40 else "bad")

    # Vol/OI poměr — neobvyklá dnešní aktivita
    vol_oi = round(volume / open_interest, 2) if open_interest > 0 else None
    vol_oi_flag = "normal"
    if vol_oi is not None:
        if vol_oi >= 1.0:
            vol_oi_flag = "unusual"   # objem přesáhl celý OI — velmi neobvyklé
        elif vol_oi >= 0.3:
            vol_oi_flag = "elevated"  # zvýšená aktivita

    # liquidity score: combines OI and volume
    liquidity = "low"
    if open_interest >= 500 or volume >= 100:
        liquidity = "high"
    elif open_interest >= 100 or volume >= 20:
        liquidity = "med"

    # tier based on IV/RV
    tier = "ok"
    if iv_rv is not None:
        if iv_rv >= rich_thresh:
            tier = "rich"
        elif iv_rv >= 1 + (rich_thresh - 1) * 0.5:
            tier = "warm"
    if roc_annual >= 0.6:
        tier = "rich" if tier != "ok" else "warm"

    # bonus signálový bod za RV Rank a Vol/OI
    bonus = 0
    if rv_rank is not None and rv_rank >= 70:
        bonus += 1   # IV je historicky vysoká
    if vol_oi_flag in ("elevated", "unusual"):
        bonus += 1   # neobvyklá poptávka dnes

    # overall signal
    base_good = tier in ("rich", "warm")
    liquid_ok = liquidity in ("high", "med")
    spread_ok = spread_flag == "ok"
    score = sum([base_good, liquid_ok, spread_ok]) + bonus

    if base_good and liquid_ok and spread_ok and bonus >= 1:
        signal = "strong+"   # všechno + historická/poptávková potvrzení
    elif base_good and liquid_ok and spread_ok:
        signal = "strong"
    elif base_good and liquid_ok:
        signal = "ok"
    elif base_good:
        signal = "illiquid"
    else:
        signal = "pass"

    return {
        "sigma":      round(sigma * 100, 1),
        "fair":       round(fair, 2),
        "delta":      round(abs(delta), 2),
        "vega":       round(greeks.get("vega", 0), 3),
        "theta":      round(greeks.get("theta", 0), 3),
        "prob_itm":   round(prob_itm * 100, 1),
        "pop":        round(pop * 100, 1) if pop is not None else None,
        "breakeven":  round(breakeven, 2),
        "roc_annual": round(roc_annual * 100, 1),
        "iv_rv":      round(iv_rv, 2) if iv_rv else None,
        "spread_pct": round(spread_pct * 100, 1),
        "spread_flag": spread_flag,
        "liquidity":  liquidity,
        "vol_oi":     vol_oi,
        "vol_oi_flag": vol_oi_flag,
        "signal":     signal,
        "tier":       tier,
    }

# ── OSI parser ─────────────────────────────────────────────────────────────────

def parse_osi(sym):
    sym = (sym or "").replace(" ", "")
    m = re.match(r"([A-Z.]+?)(\d{6})([CP])(\d{8})", sym)
    if not m:
        return None
    ymd = m.group(2)
    return {
        "cp":     m.group(3),
        "strike": int(m.group(4)) / 1000,
        "exp":    datetime(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]), tzinfo=timezone.utc),
    }

# ── Data fetchers ──────────────────────────────────────────────────────────────

async def fetch_spot_and_realized(ticker, client):
    # 1 rok dat — stačí pro RV Rank i 30d realized vol
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=365d"
    r = await client.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    result = data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    closes = [c for c in closes if c is not None]
    if len(closes) < 5:
        raise ValueError("Nedostatek historických dat")
    spot = closes[-1]

    # 30-day realized vol (poslední měsíc)
    c30 = closes[-31:] if len(closes) >= 31 else closes
    lr30 = [math.log(c30[i] / c30[i-1]) for i in range(1, len(c30))]
    rv30 = statistics.stdev(lr30) * math.sqrt(252)

    # RV Rank — kde je dnešní 30d RV v kontextu posledního roku
    # Počítáme rolling 21d RV pro každý den a zjistíme percentil
    rv_rank = None
    if len(closes) >= 52:
        window = 21
        rolling_rvs = []
        for i in range(window, len(closes)):
            chunk = closes[i-window:i+1]
            lr = [math.log(chunk[j] / chunk[j-1]) for j in range(1, len(chunk))]
            rolling_rvs.append(statistics.stdev(lr) * math.sqrt(252))
        if rolling_rvs:
            below = sum(1 for v in rolling_rvs if v <= rv30)
            rv_rank = round(below / len(rolling_rvs) * 100, 0)

    # fetch earnings date from Yahoo summary
    earnings_date = None
    try:
        url2 = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
        r2 = await client.get(url2, headers=HEADERS)
        meta = r2.json()["chart"]["result"][0]["meta"]
        ed = meta.get("earningsTimestamp") or meta.get("earningsTimestampStart")
        if ed:
            earnings_date = datetime.fromtimestamp(ed, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        pass

    return spot, rv30, rv_rank, earnings_date

async def fetch_cboe_chain(ticker, client):
    urls = [
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json",
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker}.json",
    ]
    for url in urls:
        try:
            r = await client.get(url, headers=HEADERS)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    raise ValueError("CBOE: ticker nenalezen")

# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/scan/{ticker}")
async def scan(ticker: str, r: float = 0.04, thresh: float = 25.0, exp: str = "",
               min_oi: int = 50, max_spread: float = 40.0):
    ticker = ticker.upper().strip()
    rich_thresh = 1 + thresh / 100

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        try:
            spot, realized, rv_rank, earnings_date = await fetch_spot_and_realized(ticker, client)
        except Exception as e:
            raise HTTPException(502, f"Chyba při načítání ceny: {e}")
        try:
            raw = await fetch_cboe_chain(ticker, client)
        except Exception as e:
            raise HTTPException(502, f"Chyba při načítání opčního řetězce: {e}")

    data     = raw.get("data", raw)
    all_opts = data.get("options", [])
    now      = datetime.now(timezone.utc)

    # collect expirations
    exp_set: set[str] = set()
    for o in all_opts:
        meta = parse_osi(o.get("option", ""))
        if meta and meta["cp"] == "P" and meta["exp"] > now:
            exp_set.add(meta["exp"].strftime("%Y-%m-%d"))
    expirations = sorted(exp_set)
    if not expirations:
        raise HTTPException(404, "Nenalezeny žádné budoucí expirace")

    selected_exp = exp if exp in expirations else expirations[0]

    # check if earnings fall within this expiration
    earnings_warning = False
    if earnings_date and earnings_date <= selected_exp:
        earnings_warning = True

    puts = []
    for o in all_opts:
        meta = parse_osi(o.get("option", ""))
        if not meta or meta["cp"] != "P":
            continue
        if meta["exp"].strftime("%Y-%m-%d") != selected_exp:
            continue

        strike = meta["strike"]
        if strike > spot * 1.5:
            continue

        bid  = float(o.get("bid") or 0)
        ask  = float(o.get("ask") or 0)
        last = float(o.get("last_trade_price") or 0)
        theo = float(o.get("theo") or 0)
        mid  = (bid + ask) / 2 if bid and ask else (last or theo)
        if mid <= 0:
            continue

        oi     = int(o.get("open_interest") or 0)
        volume = int(o.get("volume") or 0)

        # apply filters
        if oi < min_oi:
            continue
        spread = ask - bid
        spread_pct = (spread / mid * 100) if mid > 0 else 99
        if spread_pct > max_spread:
            continue

        iv_raw = o.get("iv")
        try:
            iv_val = float(iv_raw) if iv_raw is not None else None
        except Exception:
            iv_val = None

        dte = max(1, (meta["exp"] - now).days)
        metrics = analyze_put(
            S=spot, K=strike, dte=dte, premium=mid,
            iv_raw=iv_val if iv_val and iv_val > 0 else None,
            r=r, realized=realized, rich_thresh=rich_thresh,
            open_interest=oi, volume=volume, bid=bid, ask=ask,
            rv_rank=rv_rank,
        )
        if metrics is None:
            continue

        puts.append({
            "strike":        strike,
            "bid":           round(bid, 2),
            "ask":           round(ask, 2),
            "mid":           round(mid, 2),
            "dte":           dte,
            "open_interest": oi,
            "volume":        volume,
            **metrics,
        })

    puts.sort(key=lambda x: x["strike"], reverse=True)

    return {
        "ticker":           ticker,
        "spot":             round(spot, 2),
        "realized":         round(realized * 100, 1),
        "rv_rank":          rv_rank,
        "expirations":      expirations,
        "selected_exp":     selected_exp,
        "earnings_date":    earnings_date,
        "earnings_warning": earnings_warning,
        "puts":             puts,
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
