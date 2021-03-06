import pandas as pd
import numpy as np
import pyupbit
import matplotlib.pyplot as plt
import datetime
import pytz


class CoinInstrument:
    KST = datetime.timezone(datetime.timedelta(hours=9))

    def __init__(self, ticker, to=None, count=200, interval='day', period=0.5):
        self._data = None
        self.ticker = ticker
        self.interval = interval
        self.count = count
        self.to = to
        self.period = period
        self.load_data()
        self._data["log_returns"] = self.log_returns()

    def __repr__(self):
        return "CoinInstrument(ticker={}, to={}, count={}, interval={})".format(self.ticker, self.to,
                                                                                self.count, self.interval)

    # def to_origin(self):
    #     return self._utc_to_kst(self._to)
    #
    # def _kst_to_utc(self, to):
    #     return pd.to_datetime(to).to_pydatetime().replace(tzinfo=CoinInstrument.KST).astimezone(
    #         datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    #
    # def _utc_to_kst(self, to):
    #     return pd.to_datetime(to).to_pydatetime().replace(tzinfo=datetime.timezone.utc).astimezone(
    #         CoinInstrument.KST).strftime("%Y-%m-%d %H:%M:%S")

    def load_data(self):
        self._data = pyupbit.get_ohlcv(self.ticker, self.interval, self.count, self.to, self.period)
        # self._data.index = self._data.index.tz_convert(None).tz_localize('Asia/Seoul')

    def symbol(self):
        return self.ticker

    def log_returns(self):
        return np.log(self._data.close / self._data.close.shift(1))

    def plot_prices(self, figsize=(12, 8)):
        self._data.close.plot(figsize=figsize)
        plt.title("Close chart : {}".format(self.ticker), fontsize=15)

    def plot_returns(self, kind="ts", figsize=(12, 8)):
        if kind == "ts":
            self._data.log_returns.plot(figsize=figsize)
            plt.title("Returns : {}".format(self.ticker), fontsize=15)
        elif kind == "hist":
            self._data.log_returns.hist(figsize=figsize, bins=int(np.sqrt(len(self._data))))
            plt.title("Frequency of Returns : {}".format(self.ticker), fontsize=15)

    def head(self, n=5):
        return self._data.head(n)

    def tail(self, n=5):
        return self._data.tail(n)
    
    def sma(self, window):
        return self._data.close.rolling(window).mean()
    
    def ema(self, window):
        return self._data.close.ewm(span=window, min_periods=window).mean()

    def macd(self, ema_s, ema_l, signal_window):
        raw = pd.DataFrame(index=self._data.index)
        raw["EMA_S"] = self.ema(ema_s)  # short ema
        raw["EMA_L"] = self.ema(ema_l)  # long ema
        raw["MACD"] = raw.EMA_S - raw.EMA_L  # short-long diff
        raw["MACD_SIGNAL"] = raw.MACD.ewm(span=signal_window, min_periods=signal_window).mean()  # MACD ema
        return raw.loc[:, ["MACD", "MACD_SIGNAL"]].copy()

    def so(self, periods=14, d_window=3):
        raw = pd.DataFrame(index=self._data.index)
        raw["roll_low"] = self._data.low.rolling(periods).min()  # ??????????????? ?????????(min)
        raw["roll_high"] = self._data.high.rolling(periods).max()  # ??????????????? ?????????(max)
        raw["K"] = (self._data.close - raw.roll_low) / (raw.roll_high - raw.roll_low) * 100  # (price-min)/(max-min)*100
        raw["D"] = raw.K.rolling(d_window).mean()
        return raw.loc[:, ["K", "D"]].copy()
    
    def rsi(self, periods=14):
        raw = pd.DataFrame(index=self._data.index)
        raw["U"] = np.where(self._data.close.diff() > 0, self._data.close.diff(), 0)  # ?????? ?????? ?????????(U)
        raw["D"] = np.where(self._data.close.diff() < 0, -self._data.close.diff(), 0)  # ?????? ?????? ?????????(D)
        raw["MA_U"] = raw.U.rolling(periods).mean()  # U??? SMA(U_SMA)
        raw["MA_D"] = raw.D.rolling(periods).mean()  # D??? SMA(D_SMA)
        raw["RSI"] = raw.MA_U / (raw.MA_U + raw.MA_D) * 100  # U_SMA/(U_SMA + D_SMA)
        return raw.loc[:, ["RSI"]].copy()