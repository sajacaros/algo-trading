from algobot.CoinInstrument import CoinInstrument
from scipy.optimize import brute
import numpy as np
import pandas as pd


class SMABackTester:
    def __init__(self, symbol, to=None, count=200, interval='day', period=0.5, SMA_S=50, SMA_L=200):
        self.results = None
        self._data = None
        self.coinInstrument = CoinInstrument(symbol, to, count, interval, period)
        self._SMA_S = SMA_S
        self._SMA_L = SMA_L
        self.load_data()
    
    def __repr__(self):
        return "SMABackTester(ticker={}, to={}, count={}, interval={}, SMA_S={}, SMA_L={})" \
            .format(self.coinInstrument.ticker, self.coinInstrument.to, self.coinInstrument.count,
                    self.coinInstrument.interval, self._SMA_S, self._SMA_L)
    
    def load_data(self):
        self._data = pd.DataFrame()
        self._data["SMA_S"] = self.coinInstrument.sma(self._SMA_S)
        self._data["SMA_L"] = self.coinInstrument.sma(self._SMA_L)
    
    def set_parameters(self, SMA_S=None, SMA_L=None):
        if SMA_S is not None:
            self._SMA_S = SMA_S
        if SMA_L is not None:
            self._SMA_L = SMA_L
        self.load_data()
    
    def test_strategy(self):
        data = self._data.copy()
        data["returns"] = self.coinInstrument.log_returns()
        data.dropna(inplace=True)
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_returns(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | SMA_S = {} | SMA_L = {}".format(self.coinInstrument.symbol(), self._SMA_S, self._SMA_L)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(6, 4))
    
    def update_and_run(self, SMA):
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, SMA_S_range, SMA_L_range):
        opt = brute(self.update_and_run, (SMA_S_range, SMA_L_range), finish=None)
        return opt, -self.update_and_run(opt)


class EMABackTester:
    def __init__(self, symbol, to=None, count=200, interval='day', period=0.5, ema_s=50, ema_l=100):
        self.results = None
        self._data = None
        self.coinInstrument = CoinInstrument(symbol, to, count, interval, period)
        self._EMA_S = ema_s
        self._EMA_L = ema_l
        self.load_data()
    
    def __repr__(self):
        return "EMABackTester(ticker={}, to={}, count={}, interval={}, EMA_S={}, EMA_L={})" \
            .format(self.coinInstrument.ticker, self.coinInstrument.to, self.coinInstrument.count,
                    self.coinInstrument.interval, self._EMA_S, self._EMA_L)
    
    def load_data(self):
        self._data = pd.DataFrame()
        self._data["EMA_S"] = self.coinInstrument.ema(self._EMA_S)
        self._data["EMA_L"] = self.coinInstrument.ema(self._EMA_L)
    
    def set_parameters(self, ema_s=None, ema_l=None):
        if ema_s is not None:
            self._EMA_S = ema_s
        if ema_l is not None:
            self._EMA_L = ema_l
        self.load_data()
    
    def test_strategy(self):
        data = self._data.copy()
        data["returns"] = self.coinInstrument.log_returns()
        data.dropna(inplace=True)
        data["position"] = np.where(data["EMA_S"] > data["EMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_returns(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | EMA_S = {} | EMA_L = {}".format(self.coinInstrument.symbol(), self._EMA_S, self._EMA_L)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(6, 4))
    
    def update_and_run(self, ema):
        self.set_parameters(int(ema[0]), int(ema[1]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, ema_s_range, ema_l_range):
        opt = brute(self.update_and_run, (ema_s_range, ema_l_range), finish=None)
        return opt, -self.update_and_run(opt)


class CrossSmaEmaBackTester:
    def __init__(self, symbol, to=None, count=200, interval='day', period=0.5, window=50):
        self.results = None
        self._data = None
        self.coinInstrument = CoinInstrument(symbol, to, count, interval, period)
        self._window = window
        self.load_data()
    
    def __repr__(self):
        return "EMASmaEmaBackTester(ticker={}, to={}, count={}, interval={}, window={})" \
            .format(self.coinInstrument.ticker, self.coinInstrument.to, self.coinInstrument.count,
                    self.coinInstrument.interval, self._window)
    
    def load_data(self):
        self._data = pd.DataFrame()
        self._data["SMA"] = self.coinInstrument.sma(self._window)
        self._data["EMA"] = self.coinInstrument.ema(self._window)
    
    def set_parameters(self, window=None):
        if window is not None:
            self._window = window
        self.load_data()
    
    def test_strategy(self):
        data = self._data.copy()
        data["returns"] = self.coinInstrument.log_returns()
        data.dropna(inplace=True)
        data["position"] = np.where(data["EMA"] > data["SMA"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_returns(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | CROSS SMA = {} | EMA = {}".format(self.coinInstrument.symbol(), self._window, self._window)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(6, 4))
    
    def update_and_run(self, window):
        self.set_parameters(int(window))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, window_range):
        opt = brute(self.update_and_run, [window_range], finish=None)
        return opt, -self.update_and_run(opt)


class MACDBackTester:
    def __init__(self, symbol=None, to=None, count=200, interval='day', period=0.5, ema_s=12, ema_l=26, signal_window=9,
                 coinInstrument=None):
        self.results = None
        self._data = None
        if coinInstrument:
            self.coinInstrument = coinInstrument
        else:
            self.coinInstrument = CoinInstrument(symbol, to, count, interval, period)
        self._ema_s = ema_s
        self._ema_l = ema_l
        self._signal_window = signal_window
        self.load_data()
    
    def __repr__(self):
        return "MACDBackTester(ticker={}, to={}, count={}, interval={}, ema_s={}, ema_l={}, signal={})" \
            .format(self.coinInstrument.ticker, self.coinInstrument.to, self.coinInstrument.count,
                    self.coinInstrument.interval,
                    self._ema_s, self._ema_l, self._signal_window)
    
    def load_data(self):
        self._data = self.coinInstrument.macd(self._ema_s, self._ema_l, self._signal_window)
    
    def set_parameters(self, ema_s=None, ema_l=None, signal_window=None):
        if ema_s is not None:
            self._ema_s = ema_s
        if ema_l is not None:
            self._ema_l = ema_l
        if signal_window is not None:
            self._signal_window = signal_window
        
        self.load_data()
    
    def test_strategy(self):
        data = self._data.copy()
        data["returns"] = self.coinInstrument.log_returns()
        data.dropna(inplace=True)
        data["position"] = np.where(data["MACD"] > data["MACD_SIGNAL"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_returns(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | MACD = {}/{} | SIGNAL = {}".format(self.coinInstrument.symbol(), self._ema_s, self._ema_l,
                                                             self._signal_window)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(6, 4))
    
    def update_and_run(self, macd=None):
        self.set_parameters(int(macd[0]), int(macd[1]), int(macd[2]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, ema_s_range, ema_l_range, window_range):
        opt = brute(self.update_and_run, (ema_s_range, ema_l_range, window_range), finish=None)
        return opt, -self.update_and_run(opt)


class SOBackTester:
    def __init__(self, symbol, to=None, count=200, interval='day', period=0.5, periods=14, d_window=3):
        self.results = None
        self._data = None
        self.coinInstrument = CoinInstrument(symbol, to, count, interval, period)
        self._periods = periods
        self._d_window = d_window
        self.load_data()
    
    def __repr__(self):
        return "SOBackTester(ticker={}, to={}, count={}, interval={}, periods={}, d_window={})" \
            .format(self.coinInstrument.ticker, self.coinInstrument.to, self.coinInstrument.count,
                    self.coinInstrument.interval,
                    self._periods, self._d_window)
    
    def load_data(self):
        self._data = self.coinInstrument.so(self._periods, self._d_window)
    
    def set_parameters(self, periods=None, d_window=None):
        if periods is not None:
            self._periods = periods
        if d_window is not None:
            self._d_window = d_window
        
        self.load_data()
    
    def test_strategy(self):
        data = self._data.copy()
        data["returns"] = self.coinInstrument.log_returns()
        data.dropna(inplace=True)
        data["position"] = np.where(data["K"] > data["D"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_returns(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | SO = {}/{}".format(self.coinInstrument.symbol(), self._periods, self._d_window)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(6, 4))
    
    def update_and_run(self, so=None):
        self.set_parameters(int(so[0]), int(so[1]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, periods_range, d_window_range):
        opt = brute(self.update_and_run, (periods_range, d_window_range), finish=None)
        return opt, -self.update_and_run(opt)


class SOHorizonBackTester:
    def __init__(self, symbol, to=None, count=200, interval='day', period=0.5, periods=14, upper=80, lower=20):
        self.results = None
        self._data = None
        self.coinInstrument = CoinInstrument(symbol, to, count, interval, period)
        self._periods = periods
        self._upper = upper
        self._lower = lower
        self.load_data()
    
    def __repr__(self):
        return "SOHorizonBackTester(ticker={}, to={}, count={}, interval={}, periods={}, upper={}, lower={})" \
            .format(self.coinInstrument.ticker, self.coinInstrument.to, self.coinInstrument.count,
                    self.coinInstrument.interval,
                    self._periods, self._upper, self._lower)
    
    def load_data(self):
        self._data = self.coinInstrument.so(self._periods)
    
    def set_parameters(self, periods=None, upper=None, lower=None):
        if periods is not None:
            self._periods = periods
            self.load_data()
        if upper is not None:
            self._upper = upper
        if lower is not None:
            self._lower = lower
    
    def test_strategy(self):
        data = self._data.copy()
        data["returns"] = self.coinInstrument.log_returns()
        data["before_k"] = data.K.shift(1)
        data.dropna(inplace=True)
        # data["position"] = np.where((data["before_k"] > self._upper) & (data["K"] < self._upper), -1, np.nan)
        data["position"] = np.where((data["before_k"] > self._upper) & (data["K"] > self._upper), 0, np.nan)
        data["position"] = np.where((data["before_k"] < self._lower) & (data["K"] < self._lower), 1, data.position)
        data.position = data.position.fillna(method='ffill')
        data.position = data.position.fillna(0)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_returns(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | SO = {}".format(self.coinInstrument.symbol(), self._periods)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(6, 4))
    
    def update_and_run(self, so_horizon=None):
        self.set_parameters(int(so_horizon[0]), int(so_horizon[1]), int(so_horizon[2]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, periods_range, so_upper_range, so_lower_range):
        opt = brute(self.update_and_run, (periods_range, so_upper_range, so_lower_range), finish=None)
        return opt, -self.update_and_run(opt)


class RSIBackTester:
    def __init__(self, symbol=None, to=None, count=200, interval='day', period=0.5, periods=14, upper=80, lower=20,
                 coinInstrument=None):
        self.results = None
        self._data = None
        if coinInstrument:
            self.coinInstrument = coinInstrument
        else:
            self.coinInstrument = CoinInstrument(symbol, to, count, interval, period)
        self._periods = periods
        self._upper = upper
        self._lower = lower
        self.load_data()
    
    def __repr__(self):
        return "RSIBackTester(ticker={}, to={}, count={}, interval={}, periods={}, upper={}, lower={})" \
            .format(self.coinInstrument.ticker, self.coinInstrument.to, self.coinInstrument.count,
                    self.coinInstrument.interval,
                    self._periods, self._upper, self._lower)
    
    def load_data(self):
        self._data = self.coinInstrument.rsi(self._periods)
    
    def set_parameters(self, periods=None, upper=None, lower=None):
        if periods is not None:
            self._periods = periods
            self.load_data()
        if upper is not None:
            self._upper = upper
        if lower is not None:
            self._lower = lower
    
    def test_strategy(self):
        data = self._data.copy()
        data["returns"] = self.coinInstrument.log_returns()
        data.dropna(inplace=True)
        data["position"] = np.where(data["RSI"] > self._upper, -1, np.nan)
        data["position"] = np.where(data["RSI"] < self._lower, 1, data.position)
        data.position = data.position.fillna(0)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_returns(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | SO = {}".format(self.coinInstrument.symbol(), self._periods)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(6, 4))
    
    def update_and_run(self, so_horizon=None):
        self.set_parameters(int(so_horizon[0]), int(so_horizon[1]), int(so_horizon[2]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, periods_range, rsi_upper_range, rsi_lower_range):
        opt = brute(self.update_and_run, (periods_range, rsi_upper_range, rsi_lower_range), finish=None)
        return opt, -self.update_and_run(opt)


class CombineMACDRSIBackTester:
    def __init__(self, symbol=None, to=None, count=200, interval='day', period=0.5, macd_ema_s=12, macd_ema_l=26,
                 macd_signal_window=9, rsi_periods=14, rsi_upper=80, rsi_lower=20, coinInstrument=None):
        self.results = None
        self._data = None
        if coinInstrument:
            self.coinInstrument = coinInstrument
        else:
            self.coinInstrument = CoinInstrument(symbol, to, count, interval, period)
        self._rsi_periods = rsi_periods
        self._rsi_upper = rsi_upper
        self._rsi_lower = rsi_lower
        self._macd_ema_s = macd_ema_s
        self._macd_ema_l = macd_ema_l
        self._macd_signal_window = macd_signal_window
        self.load_data()
    
    def __repr__(self):
        return "CombineMACDRSIBackTester(RSI={}, MACD={})".format(self.rsiBackTester, self.macdBackTester)
    
    def load_data(self):
        rsiBackTester = RSIBackTester(coinInstrument=self.coinInstrument, periods=self._rsi_periods, upper=self._rsi_upper, lower=self._rsi_lower)
        macdBackTester = MACDBackTester(coinInstrument=self.coinInstrument, ema_s=self._macd_ema_s, ema_l=self._macd_ema_l, signal_window=self._macd_signal_window)
        rsiBackTester.test_strategy()
        macdBackTester.test_strategy()
        raw = pd.DataFrame(index=rsiBackTester.results.index)
        raw["MACD_POSITION"] = rsiBackTester.results.position
        raw["RSI_POSITION"] = macdBackTester.results.position
        self._data = raw
    
    def set_parameters(self, macd_ema_s=None, macd_ema_l=None, macd_signal_window=None, rsi_periods=None, rsi_upper=None, rsi_lower=None):
        if macd_ema_s is not None:
            self._macd_ema_s = macd_ema_s
        if macd_ema_l is not None:
            self._macd_ema_l = macd_ema_l
        if macd_signal_window is not None:
            self._macd_signal_window = macd_signal_window
        if rsi_periods is not None:
            self._rsi_periods = rsi_periods
        if rsi_upper is not None:
            self._rsi_upper = rsi_upper
        if rsi_lower is not None:
            self._rsi_lower = rsi_lower
        self.load_data()
    
    def test_strategy(self):
        data = self._data.copy()
        data["returns"] = self.coinInstrument.log_returns()
        data.dropna(inplace=True)
        data["position"] = np.where(data["MACD_POSITION"] == data["RSI_POSITION"], data["MACD_POSITION"], 0)
        data.position = data.position.fillna(0)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_returns(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | SO = {}".format(self.coinInstrument.symbol(), self._periods)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(6, 4))
    
    def update_and_run(self, parameter=None):
        self.set_parameters(int(parameter[0]), int(parameter[1]), int(parameter[2]), int(parameter[3]), int(parameter[4]), int(parameter[5]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, ema_s_range, ema_l_range, signal_window_range, rsi_periods_range, rsi_upper_range, rsi_lower_range):
        opt = brute(self.update_and_run, (ema_s_range, ema_l_range, signal_window_range, rsi_periods_range, rsi_upper_range, rsi_lower_range), finish=None)
        return opt, -self.update_and_run(opt)
