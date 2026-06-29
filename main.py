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

def erf(x: float) -> float:
    s = -1 if x < 0 else 1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*math.exp(-x*x)
    return s * y

def N(x: float) -> float:
    return 0.5 * (1 + erf(x / math.sqrt(2)))

def bs_put(S, K, T, r, sig) -> float:
    if T <= 0 or sig <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5*sig*sig)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    return K*math.exp(-r*T)*N(-d2) - S*N(-d1)

def solve_iv(price, S, K, T, r) -> float | None:
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

def analyze_put(S, K, dte, premium, iv_raw, r, realized, rich_thresh):
    T = dte / 365
    sigma = iv_raw
    if (sigma is None or sigma <= 0) and premium > 0:
        sigma = solve_iv(premium, S, K, T, r)
    if sigma is None or sigma <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    delta = -N(-d1)
    prob_itm = N(-d2)
    fair = bs_put(S, K, T, r, sigma)
    breakeven = K - premium
    pop = None
    if breakeven > 0:
        dbe = (math.log(S / breakeven) + (r - 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
        pop = N(dbe)
    roc_annual = (premium / K) * (365 / dte) if dte > 0 else 0
    touch = min(1.0, 2 * prob_itm)
    iv_rv = sigma / realized if realized > 0 else None
    tier = "ok"
    if iv_rv is not None:
        if iv_rv >= rich_thresh:
            tier = "rich"
        elif iv_rv >= 1 + (rich_thresh - 1) * 0.5:
            tier = "warm"
    if roc_annual >= 0.6:
        tier = "rich" if tier != "ok" else "warm"
    return {
        "sigma": round(sigma * 100, 1),
        "fair": round(fair, 2),
        "delta": round(abs(delta), 2),
        "prob_itm": round(prob_itm * 100, 1),
        "pop": round(pop * 100, 1) if pop is not None else None,
        "breakeven": round(breakeven, 2),
        "roc_annual": round(roc_annual * 100, 1),
        "touch": round(touch * 100, 1),
        "iv_rv": round(iv_rv, 2) if iv_rv else None,
        "tier": tier,
    }

# ── OSI symbol parser ──────────────────────────────────────────────────────────

def parse_osi(sym: str):
    sym = (sym or "").replace(" ", "")
    m = re.match(r"([A-Z.]+?)(\d{6})([CP])(\d{8})", sym)
    if not m:
        return None
    ymd = m.group(2)
    cp = m.group(3)
    strike = int(m.group(4)) / 1000
    exp = datetime(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]), tzinfo=timezone.utc)
    return {"cp": cp, "strike": strike, "exp": exp}

# ── Data fetchers ──────────────────────────────────────────────────────────────

async def fetch_spot_and_realized(ticker: str, client: httpx.AsyncClient):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=40d"
    r = await client.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    closes = [c for c in closes if c is not None]
    if len(closes) < 5:
        raise ValueError("Nedostatek historických dat")
    spot = closes[-1]
    log_returns = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
    realized = statistics.stdev(log_returns) * math.sqrt(252)
    return spot, realized

async def fetch_cboe_chain(ticker: str, client: httpx.AsyncClient):
    urls = [
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json",
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker}.json",
    ]
    last_err = None
    for url in urls:
        try:
            r = await client.get(url, headers=HEADERS)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            last_err = e
    raise ValueError(f"CBOE: ticker nenalezen ({last_err})")

# ── API endpoint ───────────────────────────────────────────────────────────────

@app.get("/api/scan/{ticker}")
async def scan(ticker: str, r: float = 0.04, thresh: float = 25.0, exp: str = ""):
    ticker = ticker.upper().strip()
    rich_thresh = 1 + thresh / 100

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        try:
            spot, realized = await fetch_spot_and_realized(ticker, client)
        except Exception as e:
            raise HTTPException(502, f"Chyba při načítání ceny: {e}")

        try:
            raw = await fetch_cboe_chain(ticker, client)
        except Exception as e:
            raise HTTPException(502, f"Chyba při načítání opčního řetězce: {e}")

    data = raw.get("data", raw)
    all_opts = data.get("options", [])
    now = datetime.now(timezone.utc)

    # collect unique expirations from put symbols
    exp_set: set[str] = set()
    for o in all_opts:
        sym = o.get("option", "")
        meta = parse_osi(sym)
        if meta and meta["cp"] == "P" and meta["exp"] > now:
            exp_set.add(meta["exp"].strftime("%Y-%m-%d"))
    expirations = sorted(exp_set)

    if not expirations:
        raise HTTPException(404, "Nenalezeny žádné budoucí expirace")

    selected_exp = exp if exp in expirations else expirations[0]

    # filter puts for selected expiration
    puts = []
    for o in all_opts:
        sym = o.get("option", "")
        meta = parse_osi(sym)
        if not meta or meta["cp"] != "P":
            continue
        if meta["exp"].strftime("%Y-%m-%d") != selected_exp:
            continue
        strike = meta["strike"]
        if strike > spot * 1.5:  # odřízni extrémně vzdálené strikes (>50% nad spot)
            continue
        bid = float(o.get("bid") or 0)
        ask = float(o.get("ask") or 0)
        last = float(o.get("last_trade_price") or 0)
        theo = float(o.get("theo") or 0)
        mid = (bid + ask) / 2 if bid and ask else (last or theo)
        if mid <= 0:
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
        )
        if metrics is None:
            continue
        puts.append({
            "strike": strike,
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "mid": round(mid, 2),
            "dte": dte,
            **metrics,
        })

    puts.sort(key=lambda x: x["strike"], reverse=True)

    return {
        "ticker": ticker,
        "spot": round(spot, 2),
        "realized": round(realized * 100, 1),
        "expirations": expirations,
        "selected_exp": selected_exp,
        "puts": puts,
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
