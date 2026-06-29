import httpx, json, math, statistics

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120"}

# Test Yahoo v8 chart (no auth needed)
r = httpx.get("https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d&range=40d", headers=HEADERS)
print("chart status:", r.status_code)
if r.status_code == 200:
    d = r.json()
    closes = d["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    closes = [c for c in closes if c is not None]
    print(f"spot: {closes[-1]:.2f}, {len(closes)} closes")

# Test CBOE options
r2 = httpx.get("https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json", headers=HEADERS)
print("CBOE status:", r2.status_code)
if r2.status_code == 200:
    d2 = r2.json()
    data = d2.get("data", d2)
    opts = data.get("options", [])
    puts = [o for o in opts if "P" in (o.get("option","") or "")]
    print(f"puts: {len(puts)}, sample: {puts[:2]}")
