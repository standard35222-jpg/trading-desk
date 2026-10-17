#!/usr/bin/env python3
"""
trend_trader.py — プロの手法を組み込んだ売買判断＆バックテストツール

設計の考え方（前回の相談内容をそのままルール化）
  1. コア＋サテライト：資産の一部は常に指数ETFを保有（上昇相場で置いていかれない）
  2. 相場環境フィルター：指数が200日線より上のときだけ新規買い
  3. 銘柄ごとに「トレンド」か「レンジ」かを判定して戦い方を切り替える
       トレンド … Minervini流トレンドテンプレート → 20日高値ブレイク or 20日線への押し目
       レンジ   … ボックス下限付近で陽線 → 下限割れで撤退、上限手前で利確
  4. サイズは損失額から逆算（1トレードの損失＝資産の1%）
  5. 出口を先に決める：ATRストップ＋トレーリング、50日線割れ、レンジは時間切れ
  6. 現実に寄せた検証：終値で判断→翌日寄付で約定、手数料・スリッページ・年末の税金20.315%

使い方
  pip install yfinance pandas numpy matplotlib
  python trend_trader.py            # バックテスト＋明日の注文候補
  python trend_trader.py --demo     # ネットなしで動作確認（合成データ）
  python trend_trader.py --refresh  # 株価データを取り直す

※ 売買判断の補助ツールです。発注は自分で行ってください。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# ============================================================
# 設定（ここを書き換えて調整）
# ============================================================
CONFIG = dict(
    capital=10_000_000,          # 初期資金（円）
    core_ticker="1306.T",        # コアで常に持つ指数ETF（TOPIX）
    compare=["1306.T", "2644.T"],  # 比較対象（TOPIX、半導体ETF）
    universe=[                   # サテライトで売買する銘柄
        "6525.T", "4062.T", "6857.T", "5801.T", "5802.T", "6834.T",
        "6278.T", "6134.T", "6941.T", "3110.T", "4203.T", "6752.T",
        "8316.T", "6146.T", "6920.T", "6871.T", "7011.T", "8725.T",
    ],
    years=10,                    # 取得する過去データの年数
    core_weight=0.5,             # コア比率（0〜1）
    risk_per_trade=0.01,         # 1トレードで失ってよい額（資産比）
    max_pos_pct=0.15,            # 1銘柄の上限（資産比）
    max_positions=6,             # サテライトの最大保有数
    commission=0.0005,           # 片道手数料
    slippage=0.001,              # 片道スリッページ
    tax_rate=0.20315,            # 年間の実現益に課税
    atr_stop=2.0,                # 初期ストップ＝エントリー − 2ATR
    atr_trail=3.0,               # トレーリング＝保有中の高値 − 3ATR
    range_time_stop=15,          # レンジトレードの最大保有日数
    min_rr=2.0,                  # レンジトレードの最低リスクリワード
)


# ============================================================
# データ取得
# ============================================================
def load_yahoo(tickers, years, cache_dir="data", refresh=False):
    """yfinanceで取得しCSVにキャッシュ。start=のバグを避けるため period= を使う"""
    import yfinance as yf
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for t in tickers:
        path = os.path.join(cache_dir, f"{t}.csv")
        fresh = os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600
        if fresh and not refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = yf.Ticker(t).history(period=f"{years}y", auto_adjust=True)
            if df.empty:
                print(f"  ! {t}: データ取得失敗、スキップ")
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            df.to_csv(path)
        out[t] = df.dropna()
    return out


def make_demo(tickers, years=8, seed=7):
    """トレンドとレンジが入れ替わる合成データ（動作確認用）"""
    rng = np.random.default_rng(seed)
    n = int(252 * years)
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    out = {}
    for k, t in enumerate(tickers):
        drift = np.zeros(n)
        i = 0
        while i < n:
            L = rng.integers(60, 250)
            drift[i:i + L] = rng.choice([0.0012, 0.0, -0.0008], p=[0.45, 0.35, 0.2])
            i += L
        vol = 0.012 if k == 0 else rng.uniform(0.018, 0.03)
        ret = drift + rng.normal(0, vol, n)
        close = (1000 if k else 2000) * np.exp(np.cumsum(ret))
        op = close * np.exp(rng.normal(0, vol / 3, n))
        hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
        lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
        v = rng.lognormal(12, 0.4, n)
        out[t] = pd.DataFrame(dict(Open=op, High=hi, Low=lo, Close=close, Volume=v), index=idx)
    return out


# ============================================================
# 指標
# ============================================================
def add_indicators(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    for n in (20, 50, 150, 200):
        d[f"sma{n}"] = c.rolling(n).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # ADX（トレンドの強さ。20未満ならレンジ寄り）
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / d["atr"]
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / d["atr"]
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    d["hh20"] = h.rolling(20).max().shift(1)          # 前日までの20日高値
    d["hi252"] = h.rolling(252, min_periods=120).max()
    d["lo252"] = l.rolling(252, min_periods=120).min()
    d["box_hi"] = h.rolling(40).max()
    d["box_lo"] = l.rolling(40).min()
    d["vol50"] = d["Volume"].rolling(50).mean()

    # 相対強度（IBD風：直近3ヶ月を2倍に重み付け）
    r = lambda n: c / c.shift(n) - 1
    d["rs"] = 2 * r(63) + r(126) + r(189) + r(252)

    # Minervini流トレンドテンプレート
    d["trend_ok"] = (
        (c > d["sma50"]) & (d["sma50"] > d["sma150"]) & (d["sma150"] > d["sma200"])
        & (d["sma200"] > d["sma200"].shift(20))
        & (c >= 0.75 * d["hi252"]) & (c >= 1.3 * d["lo252"])
    )
    # レンジ判定：トレンドが弱い＆50日線が横ばい＆箱の幅が8〜30%
    flat = (d["sma50"] / d["sma50"].shift(20) - 1).abs() < 0.03
    width = (d["box_hi"] - d["box_lo"]) / d["box_lo"]
    d["range_ok"] = (d["adx"] < 20) & flat & width.between(0.08, 0.30)
    return d


# ============================================================
# バックテスト
# ============================================================
def lot_of(ticker):
    return 100 if ticker.endswith(".T") else 1


class Backtester:
    def __init__(self, raw, cfg):
        self.cfg = cfg
        core = cfg["core_ticker"]
        self.dates = raw[core].index
        self.d = {t: add_indicators(df).reindex(self.dates) for t, df in raw.items()}
        self.universe = [t for t in cfg["universe"] if t in self.d]
        self.cash = float(cfg["capital"])
        self.core_sh = 0
        self.core_cost = 0.0
        self.pos = {}        # ticker -> dict
        self.pending = []    # 翌日寄付で執行する注文
        self.trades = []
        self.realized = {}   # 年 -> 実現損益
        self.taxes = 0.0
        self.equity = []
        self.last_px = {}

    # ---------- 共通 ----------
    def _fee(self, value):
        return value * self.cfg["commission"]

    def _book(self, date, pnl):
        self.realized[date.year] = self.realized.get(date.year, 0.0) + pnl

    def _sell(self, t, date, price, reason):
        p = self.pos.pop(t)
        px = price * (1 - self.cfg["slippage"])
        value = px * p["sh"]
        fee = self._fee(value)
        self.cash += value - fee
        pnl = value - fee - p["cost"]
        self._book(date, pnl)
        risk = (p["entry"] - p["init_stop"]) * p["sh"]
        self.trades.append(dict(
            ticker=t, setup=p["setup"], entry_date=p["date"].date(), exit_date=date.date(),
            entry=round(p["entry"], 1), exit=round(px, 1), shares=p["sh"],
            pnl=round(pnl), ret_pct=round(100 * pnl / p["cost"], 2),
            R=round(pnl / risk, 2) if risk > 0 else np.nan,
            days=(date - p["date"]).days, reason=reason))

    def _equity_now(self, i):
        eq = self.cash
        for t, p in self.pos.items():
            eq += p["sh"] * self.last_px.get(t, p["entry"])
        eq += self.core_sh * self.last_px.get(self.cfg["core_ticker"], 0)
        return eq

    # ---------- メインループ ----------
    def run(self):
        cfg, core = self.cfg, self.cfg["core_ticker"]
        prev_month = None
        for i, date in enumerate(self.dates):
            # 年が変わったら前年の実現益に課税
            if i > 0 and date.year != self.dates[i - 1].year:
                gain = self.realized.get(self.dates[i - 1].year, 0.0)
                if gain > 0:
                    tax = gain * cfg["tax_rate"]
                    self.cash -= tax
                    self.taxes += tax

            # 1) 寄付で注文執行
            self._execute(i, date)

            # 2) 月初にコアをリバランス（寄付）
            if date.month != prev_month:
                self._rebalance_core(i, date)
                prev_month = date.month

            # 3) 場中：ストップ・利確
            for t in list(self.pos):
                row = self.d[t].iloc[i]
                if np.isnan(row["Close"]):
                    continue
                p = self.pos[t]
                if row["Low"] <= p["stop"]:
                    self._sell(t, date, min(row["Open"], p["stop"]), "ストップ")
                elif p["mode"] == "range" and row["High"] >= p["target"]:
                    self._sell(t, date, max(row["Open"], p["target"]), "レンジ利確")

            # 4) 終値で状態更新
            for t, df in self.d.items():
                c = df["Close"].iloc[i]
                if not np.isnan(c):
                    self.last_px[t] = c
            eq = self._equity_now(i)
            invested = sum(p["sh"] * self.last_px.get(t, p["entry"]) for t, p in self.pos.items())
            self.equity.append((date, eq, invested / eq if eq > 0 else 0))

            # 5) 終値で翌日の注文を作る
            self._signals(i, date, eq)
        return self

    def _execute(self, i, date):
        cfg = self.cfg
        orders, self.pending = self.pending, []
        for o in orders:
            row = self.d[o["ticker"]].iloc[i]
            if np.isnan(row["Open"]):
                self.pending.append(o)      # 休場などは翌日に持ち越し
                continue
            if o["side"] == "sell":
                if o["ticker"] in self.pos:
                    self._sell(o["ticker"], date, row["Open"], o["reason"])
                continue
            px = row["Open"] * (1 + cfg["slippage"])
            if o["mode"] == "range":
                stop = o["stop"]
                if px <= stop * 1.01 or px >= o["target"]:
                    continue                # 寄りで前提が崩れたら見送り
            else:
                stop = px - cfg["atr_stop"] * o["atr"]
            sh = o["sh"]
            lot = lot_of(o["ticker"])
            while sh > 0 and px * sh * (1 + cfg["commission"]) > self.cash:
                sh -= lot
            if sh <= 0:
                continue
            cost = px * sh + self._fee(px * sh)
            self.cash -= cost
            self.pos[o["ticker"]] = dict(
                sh=sh, entry=px, cost=cost, stop=stop, init_stop=stop, high=px,
                mode=o["mode"], setup=o["setup"], target=o.get("target"), date=date, held=0)

    def _rebalance_core(self, i, date):
        cfg, core = self.cfg, self.cfg["core_ticker"]
        row = self.d[core].iloc[i]
        if np.isnan(row["Open"]) or cfg["core_weight"] <= 0:
            return
        eq = self._equity_now(i)
        target_sh = int(eq * cfg["core_weight"] / row["Open"])
        diff = target_sh - self.core_sh
        if abs(diff) * row["Open"] < eq * 0.02:     # ズレ2%未満は放置（税と手数料の節約）
            return
        if diff > 0:
            px = row["Open"] * (1 + cfg["slippage"])
            diff = min(diff, int(self.cash / (px * (1 + cfg["commission"]))))
            if diff <= 0:
                return
            cost = px * diff + self._fee(px * diff)
            self.cash -= cost
            self.core_cost += cost
            self.core_sh += diff
        else:
            n = -diff
            px = row["Open"] * (1 - cfg["slippage"])
            avg = self.core_cost / self.core_sh
            value = px * n - self._fee(px * n)
            self._book(date, value - avg * n)
            self.cash += value
            self.core_cost -= avg * n
            self.core_sh -= n

    def _signals(self, i, date, eq):
        cfg, core = self.cfg, self.cfg["core_ticker"]
        # 保有銘柄の出口判定
        for t, p in self.pos.items():
            row = self.d[t].iloc[i]
            if np.isnan(row["Close"]):
                continue
            p["held"] += 1
            if p["mode"] == "range" and row["trend_ok"]:
                p["mode"], p["target"], p["setup"] = "trend", None, p["setup"] + "→トレンド"
            if p["mode"] == "trend":
                p["high"] = max(p["high"], row["High"])
                p["stop"] = max(p["stop"], p["high"] - cfg["atr_trail"] * row["atr"])
                if row["Close"] < row["sma50"]:
                    self.pending.append(dict(side="sell", ticker=t, reason="50日線割れ"))
            elif p["held"] >= cfg["range_time_stop"]:
                self.pending.append(dict(side="sell", ticker=t, reason="時間切れ"))

        # 相場環境フィルター
        crow = self.d[core].iloc[i]
        if not (crow["Close"] > crow["sma200"]):
            return
        slots = cfg["max_positions"] - len(self.pos)
        if slots <= 0:
            return

        cands = []
        for t in self.universe:
            if t in self.pos:
                continue
            r = self.d[t].iloc[i]
            if np.isnan(r["Close"]) or np.isnan(r["sma200"]) or np.isnan(r["atr"]):
                continue
            c = r["Close"]
            if r["trend_ok"]:
                vol_ok = (r["vol50"] > 0) and (r["Volume"] > 1.5 * r["vol50"])
                breakout = c > r["hh20"] and vol_ok
                pullback = r["Low"] <= r["sma20"] * 1.01 and c > r["sma20"] and c > r["Open"]
                if breakout or pullback:
                    cands.append(dict(ticker=t, mode="trend", setup="ブレイク" if breakout else "押し目",
                                      stop=c - cfg["atr_stop"] * r["atr"], target=None,
                                      atr=r["atr"], close=c, rs=r["rs"]))
            elif r["range_ok"]:
                w = r["box_hi"] - r["box_lo"]
                if c <= r["box_lo"] + 0.25 * w and c > r["Open"]:
                    stop = r["box_lo"] - 0.5 * r["atr"]
                    target = r["box_lo"] + 0.8 * w
                    if c > stop and (target - c) / (c - stop) >= cfg["min_rr"]:
                        cands.append(dict(ticker=t, mode="range", setup="レンジ下限",
                                          stop=stop, target=target, atr=r["atr"], close=c, rs=r["rs"]))

        cands.sort(key=lambda x: -x["rs"] if not np.isnan(x["rs"]) else 1e9)
        budget = self.cash
        for cnd in cands[:slots]:
            risk_ps = cnd["close"] - cnd["stop"]
            if risk_ps <= 0:
                continue
            lot = lot_of(cnd["ticker"])
            sh = eq * cfg["risk_per_trade"] / risk_ps
            sh = min(sh, eq * cfg["max_pos_pct"] / cnd["close"], budget / cnd["close"])
            sh = int(sh // lot) * lot
            if sh <= 0:
                continue
            budget -= sh * cnd["close"]
            cnd.update(side="buy", sh=sh)
            self.pending.append(cnd)


# ============================================================
# 集計・レポート
# ============================================================
def stats(series, exposure=None):
    s = series.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    ret = s.pct_change().dropna()
    dd = (s / s.cummax() - 1).min()
    out = dict(
        総リターン=f"{(s.iloc[-1] / s.iloc[0] - 1) * 100:,.1f}%",
        年率リターン=f"{((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100:.1f}%",
        最大下落率=f"{dd * 100:.1f}%",
        シャープレシオ=f"{ret.mean() / ret.std() * math.sqrt(252):.2f}" if ret.std() > 0 else "-",
    )
    if exposure is not None:
        out["サテライト平均投下率"] = f"{exposure.mean() * 100:.0f}%"
    return out


MOBILE_TEMPLATE = r"""<title>%%TITLE%%</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@500;600&display=swap">
<style>
/* 朝の注文票：上から「今日やること → 持ち玉 → 成績」の1カラム */
:root{
  --bg:#eef1f5; --card:#ffffff; --fg:#152030; --muted:#5b6a7d; --line:#d6dde6;
  --accent:#2c56c4; --up:#17805a; --down:#c3412f; --warn:#a86b12;
  --s1:#2c56c4; --s2:#8a96a8; --s3:#c98a1c;
  --sans:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0e141d; --card:#161f2b; --fg:#e4e9f1; --muted:#8c99ab; --line:#26323f;
  --accent:#7f9ffa; --up:#4cc08f; --down:#f0806f; --warn:#e0a84a;
  --s1:#7f9ffa; --s2:#6e7c8f; --s3:#e0a84a; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e141d; --card:#161f2b; --fg:#e4e9f1; --muted:#8c99ab; --line:#26323f;
  --accent:#7f9ffa; --up:#4cc08f; --down:#f0806f; --warn:#e0a84a;
  --s1:#7f9ffa; --s2:#6e7c8f; --s3:#e0a84a; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.6;margin:0}
.wrap{max-width:560px;margin:0 auto;padding-block:20px 40px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  display:flex;flex-direction:column;gap:22px}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
header{display:flex;flex-direction:column;gap:6px}
.eyebrow{font-size:12px;letter-spacing:.08em;color:var(--muted)}
h1{font-size:22px;margin:0;font-weight:700;text-wrap:balance}
.regime{display:inline-flex;align-items:center;gap:6px;align-self:flex-start;font-size:13px;font-weight:500;
  padding:3px 10px;border-radius:999px;border:1px solid currentColor}
.regime.on{color:var(--up)} .regime.off{color:var(--down)}
.regime::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.demo{font-size:13px;color:var(--warn);border:1px dashed var(--warn);border-radius:8px;padding:8px 12px}
h2{font-size:13px;letter-spacing:.08em;color:var(--muted);font-weight:500;margin:0 0 8px}
.list{display:flex;flex-direction:column;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.row{display:grid;grid-template-columns:auto 1fr auto;gap:4px 12px;align-items:center;padding:12px 14px;border-top:1px solid var(--line)}
.row:first-child{border-top:0}
.tag{font-size:12px;font-weight:700;padding:2px 8px;border-radius:6px;color:var(--card)}
.tag.buy{background:var(--up)} .tag.sell{background:var(--down)} .tag.hold{background:var(--s2)}
.tk{font-weight:700;min-width:0}
.tk small{font-weight:400;color:var(--muted);margin-left:6px;font-size:12px}
.qty{text-align:right}
.detail{grid-column:2/4;font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:2px 14px}
.detail b{color:var(--fg);font-weight:500}
.bar{grid-column:2/4;height:4px;background:var(--line);border-radius:2px;position:relative}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:2px;background:var(--accent)}
.empty{padding:14px;color:var(--muted);font-size:14px}
.chart{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;display:flex;flex-direction:column;gap:10px}
.chart svg{width:100%;height:150px;display:block}
.legend{display:flex;flex-direction:column;gap:4px;font-size:13px}
.legend div{display:grid;grid-template-columns:12px 1fr auto auto;gap:10px;align-items:center}
.legend span.sw{width:12px;height:3px;border-radius:2px}
.legend .r{text-align:right;min-width:64px}
.legend .h{color:var(--muted);font-size:11px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.stats div{background:var(--card);padding:10px 12px;display:flex;flex-direction:column}
.stats span{font-size:11px;color:var(--muted)}
.stats b{font-size:17px}
.pos{color:var(--up)} .neg{color:var(--down)}
.foot{font-size:12px;color:var(--muted)}
</style>
<div class="wrap">
<header>
  <div class="eyebrow num">%%DATE%% 終値で判定</div>
  <h1>次の寄付でやること</h1>
  <div class="regime %%REGCLS%%">%%REGTXT%%</div>
</header>
%%DEMO%%
<section><h2>注文</h2><div class="list">%%ORDERS%%</div></section>
<section><h2>システムの持ち玉</h2><div class="list">%%HOLD%%</div></section>
<section><h2>指数との比較</h2><div class="chart">%%CHART%%</div></section>
<section><h2>トレード成績</h2><div class="stats">%%STATS%%</div></section>
<section><h2>セットアップ別</h2><div class="list">%%SETUPS%%</div></section>
<p class="foot">売買判断の補助です。発注はご自身で。システムは実現益に毎年20.315%課税、持ちっぱなしは非課税扱いで比較しています。</p>
</div>
"""


def _h(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def mobile_html(bt, cfg, eq, tr, rows, curves, buys, sells):
    core = cfg["core_ticker"]
    crow = bt.d[core].iloc[-1]
    on = bool(crow["Close"] > crow["sma200"])
    reg = ("on", f"相場環境 OK：{core} が200日線の上。新規買い可") if on else \
          ("off", f"相場環境 NG：{core} が200日線の下。新規買いは停止")

    o = []
    for s in sells:
        o.append(f'<div class="row"><span class="tag sell">売り</span><span class="tk num">{_h(s["ticker"])}</span>'
                 f'<span class="qty">寄付</span><div class="detail">理由 <b>{_h(s["reason"])}</b></div></div>')
    for b in buys:
        tgt = f'<span>利確 <b class="num">{b["target"]:,.0f}</b></span>' if b["target"] else ""
        o.append(f'<div class="row"><span class="tag buy">買い</span>'
                 f'<span class="tk num">{_h(b["ticker"])}<small>{_h(b["setup"])}</small></span>'
                 f'<span class="qty num">{b["sh"]:,}株</span>'
                 f'<div class="detail"><span>目安 <b class="num">{b["close"]:,.0f}</b></span>'
                 f'<span>ストップ <b class="num">{b["stop"]:,.0f}</b></span>{tgt}</div></div>')
    orders = "".join(o) or '<div class="empty">注文なし。今日は何もしなくてOKです。</div>'

    h = []
    for t, p in bt.pos.items():
        now = bt.last_px.get(t, p["entry"])
        pl = (now / p["entry"] - 1) * 100
        # ストップ（0%）〜 取得値の2倍の距離（100%）で現在位置を表示
        span = max(p["entry"] - p["init_stop"], 1e-9) * 2
        pct = max(0, min(100, (now - p["stop"]) / span * 100))
        cls = "pos" if pl >= 0 else "neg"
        h.append(f'<div class="row"><span class="tag hold">保有</span>'
                 f'<span class="tk num">{_h(t)}<small>{_h(p["setup"])}</small></span>'
                 f'<span class="qty num {cls}">{pl:+.1f}%</span>'
                 f'<div class="detail"><span>{p["sh"]:,}株</span><span>取得 <b class="num">{p["entry"]:,.0f}</b></span>'
                 f'<span>現在 <b class="num">{now:,.0f}</b></span><span>ストップ <b class="num">{p["stop"]:,.0f}</b></span></div>'
                 f'<div class="bar" title="ストップまでの余裕"><i style="width:{pct:.0f}%"></i></div></div>')
    h.append(f'<div class="row"><span class="tag hold">コア</span><span class="tk num">{_h(core)}</span>'
             f'<span class="qty num">{bt.core_sh:,}株</span></div>')
    hold = "".join(h)

    # チャート（SVGは線だけ、数値はHTML側に出す）
    names = list(curves)
    colors = ["var(--s1)", "var(--s2)", "var(--s3)"]
    lo = min(float(s.min()) for s in curves.values())
    hi = max(float(s.max()) for s in curves.values())
    W, H = 360, 150
    lines_svg = []
    for k, n in enumerate(names):
        s = curves[n].dropna()
        step = max(1, len(s) // 240)
        s = s.iloc[::step]
        pts = " ".join(f"{i / (len(s) - 1) * W:.1f},{H - 4 - (v - lo) / (hi - lo) * (H - 8):.1f}"
                       for i, v in enumerate(s.values))
        lines_svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[k % 3]}" '
                         f'stroke-width="{2.2 if k == 0 else 1.4}" vector-effect="non-scaling-stroke" '
                         f'stroke-linejoin="round"/>')
    base_y = H - 4 - (cfg["capital"] - lo) / (hi - lo) * (H - 8)
    svg = (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="img" aria-label="資産推移の比較">'
           f'<line x1="0" x2="{W}" y1="{base_y:.1f}" y2="{base_y:.1f}" stroke="var(--line)" '
           f'stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>'
           + "".join(reversed(lines_svg)) + "</svg>")
    leg = ['<div class="h"><span></span><span></span><span class="r">年率</span><span class="r">最大下落</span></div>']
    keys = list(rows)
    for k, n in enumerate(names):
        st = rows[keys[k]]
        label = "このシステム" if k == 0 else f"{n} 持ちっぱなし"
        leg.append(f'<div><span class="sw" style="background:{colors[k % 3]}"></span><span>{_h(label)}</span>'
                   f'<span class="r num">{st["年率リターン"]}</span><span class="r num">{st["最大下落率"]}</span></div>')
    chart = (f'<div class="eyebrow num">{eq.index[0].date()} 〜 {eq.index[-1].date()}　点線＝初期資金</div>'
             + svg + '<div class="legend">' + "".join(leg) + "</div>")

    if len(tr):
        win = tr[tr.pnl > 0]
        loss = tr[tr.pnl <= 0]
        pf = win.pnl.sum() / -loss.pnl.sum() if len(loss) and loss.pnl.sum() < 0 else float("inf")
        exp = tr.R.mean()
        cells = [("取引回数", f"{len(tr)}", ""), ("勝率", f"{len(win) / len(tr) * 100:.0f}%", ""),
                 ("期待値", f"{exp:+.2f}R", "pos" if exp > 0 else "neg"),
                 ("平均利益", f"{win.ret_pct.mean():.1f}%" if len(win) else "-", "pos"),
                 ("平均損失", f"{loss.ret_pct.mean():.1f}%" if len(loss) else "-", "neg"),
                 ("PF", f"{pf:.2f}", "pos" if pf > 1 else "neg")]
        stats_html = "".join(f'<div><span>{a}</span><b class="num {c}">{b}</b></div>' for a, b, c in cells)
        g = tr.groupby("setup").agg(n=("pnl", "size"), w=("pnl", lambda x: (x > 0).mean() * 100),
                                    r=("R", "mean"), p=("pnl", "sum")).sort_values("p", ascending=False)
        su = []
        for s, r in g.iterrows():
            c = "pos" if r.p > 0 else "neg"
            su.append(f'<div class="row"><span></span><span class="tk">{_h(s)}</span>'
                      f'<span class="qty num {c}">{r.p / 10000:+,.0f}万円</span>'
                      f'<div class="detail"><span>{int(r.n)}回</span><span>勝率 <b class="num">{r.w:.0f}%</b></span>'
                      f'<span>平均 <b class="num">{r.r:+.2f}R</b></span></div></div>')
        setups = "".join(su)
    else:
        stats_html = '<div><span>取引</span><b>なし</b></div>'
        setups = '<div class="empty">まだ取引がありません</div>'

    demo = ('<div class="demo">これは合成データによる表示見本です。実際の株価ではありません。'
            'PCで <span class="num">trend_trader.py</span> を回すと本物のデータで同じ画面ができます。</div>'
            if cfg.get("demo") else "")
    rep = {
        "%%TITLE%%": "朝の注文票", "%%DATE%%": str(eq.index[-1].date()),
        "%%REGCLS%%": reg[0], "%%REGTXT%%": _h(reg[1]), "%%DEMO%%": demo,
        "%%ORDERS%%": orders, "%%HOLD%%": hold, "%%CHART%%": chart,
        "%%STATS%%": stats_html, "%%SETUPS%%": setups,
    }
    out = MOBILE_TEMPLATE
    for k, v in rep.items():
        out = out.replace(k, v)
    return out


def report(bt, raw, cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    eq = pd.DataFrame(bt.equity, columns=["date", "equity", "exposure"]).set_index("date")
    eq.to_csv(os.path.join(outdir, "equity.csv"))
    tr = pd.DataFrame(bt.trades)
    tr.to_csv(os.path.join(outdir, "trades.csv"), index=False, encoding="utf-8-sig")

    lines = []
    P = lines.append
    P("=" * 56)
    P(f" 検証期間 {eq.index[0].date()} 〜 {eq.index[-1].date()}")
    P("=" * 56)
    rows = {"このシステム": stats(eq["equity"], eq["exposure"])}
    curves = {"このシステム": eq["equity"]}
    for t in cfg["compare"]:
        if t in raw:
            c = raw[t]["Close"].reindex(eq.index).ffill()
            c = c / c.dropna().iloc[0] * cfg["capital"]
            rows[f"{t} 持ちっぱなし"] = stats(c)
            curves[t] = c
    for name, s in rows.items():
        P(f"\n[{name}]")
        for k, v in s.items():
            P(f"  {k:<14}{v}")
    P(f"\n  支払った税金（累計）  {bt.taxes:,.0f}円")
    P("  ※持ちっぱなしは未実現益なので税金ゼロ扱い。システムは実現益に毎年課税")

    if len(tr):
        win = tr[tr.pnl > 0]
        loss = tr[tr.pnl <= 0]
        pf = win.pnl.sum() / -loss.pnl.sum() if len(loss) and loss.pnl.sum() < 0 else float("inf")
        P("\n[トレード成績]")
        P(f"  取引回数        {len(tr)}")
        P(f"  勝率            {len(win) / len(tr) * 100:.0f}%")
        P(f"  平均利益        {win.ret_pct.mean():.1f}%" if len(win) else "  平均利益 -")
        P(f"  平均損失        {loss.ret_pct.mean():.1f}%" if len(loss) else "  平均損失 -")
        P(f"  プロフィットF   {pf:.2f}")
        P(f"  期待値(R)       {tr.R.mean():.2f}R  ※1回の取引で初期リスクの何倍を稼いだか")
        P("\n[セットアップ別]")
        g = tr.groupby("setup").agg(回数=("pnl", "size"), 勝率=("pnl", lambda x: (x > 0).mean() * 100),
                                    平均R=("R", "mean"), 損益=("pnl", "sum"))
        for s, r in g.iterrows():
            P(f"  {s:<12} {int(r.回数):>4}回  勝率{r.勝率:>3.0f}%  平均{r.平均R:>5.2f}R  損益{r.損益:>12,.0f}円")

    # 明日の注文
    P("\n" + "=" * 56)
    P(f" 次の営業日の寄付注文（{eq.index[-1].date()} 終値で判定）")
    P("=" * 56)
    buys = [o for o in bt.pending if o["side"] == "buy"]
    sells = [o for o in bt.pending if o["side"] == "sell"]
    if not buys and not sells:
        P("  注文なし")
    for o in sells:
        P(f"  売り  {o['ticker']:<8} 理由：{o['reason']}")
    for o in buys:
        tgt = f"  利確{o['target']:,.0f}" if o["target"] else ""
        P(f"  買い  {o['ticker']:<8} {o['sh']:>6}株  [{o['setup']}]  目安{o['close']:,.0f}  "
          f"ストップ{o['stop']:,.0f}{tgt}")
    P("\n[システムの現在の保有]")
    if not bt.pos:
        P("  なし")
    for t, p in bt.pos.items():
        P(f"  {t:<8} {p['sh']:>6}株  取得{p['entry']:,.0f}  現在{bt.last_px.get(t, 0):,.0f}  "
          f"ストップ{p['stop']:,.0f}  [{p['setup']}]")
    P(f"  コア {cfg['core_ticker']} {bt.core_sh}株")

    text = "\n".join(lines)
    with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

    # ケータイ用ページ
    frag = mobile_html(bt, cfg, eq, tr, rows, curves, buys, sells)
    with open(os.path.join(outdir, "mobile_fragment.html"), "w", encoding="utf-8") as f:
        f.write(frag)
    with open(os.path.join(outdir, "mobile.html"), "w", encoding="utf-8") as f:
        f.write('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                '<meta name="robots" content="noindex,nofollow">'
                '</head><body>' + frag + "</body></html>")
    print(f"\nケータイ用: {os.path.join(outdir, 'mobile.html')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3, 1]))
        for name, s in curves.items():
            a1.plot(s.index, s / 1e6, label="System" if name == "このシステム" else name,
                    lw=2 if name == "このシステム" else 1.2)
        a1.set_ylabel("Equity (M JPY)")
        a1.legend()
        a1.grid(alpha=0.3)
        dd = eq["equity"] / eq["equity"].cummax() - 1
        a2.fill_between(dd.index, dd * 100, 0, alpha=0.4)
        a2.set_ylabel("Drawdown %")
        a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "equity.png"), dpi=120)
        print(f"\nグラフ: {os.path.join(outdir, 'equity.png')}")
    except Exception as e:
        print(f"(グラフ作成スキップ: {e})")


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="プロの手法を組み込んだ売買判断＆バックテスト")
    ap.add_argument("--demo", action="store_true", help="合成データで動作確認")
    ap.add_argument("--refresh", action="store_true", help="株価データを再取得")
    ap.add_argument("--capital", type=float)
    ap.add_argument("--core-weight", type=float)
    ap.add_argument("--years", type=int)
    ap.add_argument("--out", default="output")
    ap.add_argument("--seed", type=int, default=7, help="デモ用の乱数シード")
    a = ap.parse_args()

    cfg = dict(CONFIG)
    if a.capital: cfg["capital"] = a.capital
    if a.core_weight is not None: cfg["core_weight"] = a.core_weight
    if a.years: cfg["years"] = a.years

    tickers = list(dict.fromkeys([cfg["core_ticker"], *cfg["compare"], *cfg["universe"]]))
    if a.demo:
        cfg["demo"] = True
        raw = make_demo(tickers, seed=a.seed)
    else:
        print("株価データ取得中…")
        raw = load_yahoo(tickers, cfg["years"], refresh=a.refresh)
    if cfg["core_ticker"] not in raw:
        sys.exit("コアETFのデータが取れませんでした")

    bt = Backtester(raw, cfg).run()
    report(bt, raw, cfg, a.out)


if __name__ == "__main__":
    main()
