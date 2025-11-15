"""
技术指标计算模块 - 计算各类技术分析指标
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """技术指标计算器"""

    def __init__(self, df: pd.DataFrame):
        """
        初始化
        Args:
            df: 包含OHLCV数据的DataFrame
        """
        self.df = df.copy()

    def calculate_all_indicators(self, config: dict = None) -> pd.DataFrame:
        """
        计算所有技术指标

        Args:
            config: 指标参数配置

        Returns:
            添加了所有指标的DataFrame
        """
        if config is None:
            config = self._get_default_config()

        logger.info("开始计算技术指标")

        # 趋势类指标
        self.df = self.calculate_ema(self.df, config['ema_periods'])
        self.df = self.calculate_macd(
            self.df,
            config['macd_fast'],
            config['macd_slow'],
            config['macd_signal']
        )
        self.df = self.calculate_bollinger_bands(
            self.df,
            config['bb_period'],
            config['bb_std']
        )

        # 动量类指标
        self.df = self.calculate_rsi(self.df, config['rsi_period'])
        self.df = self.calculate_kdj(
            self.df,
            config['kdj_period'],
            config['kdj_smooth_k'],
            config['kdj_smooth_d']
        )
        self.df = self.calculate_williams_r(self.df, config['williams_period'])

        # 成交量指标
        self.df = self.calculate_obv(self.df)
        self.df = self.calculate_volume_ma(self.df, config['volume_ma_period'])

        # 支撑阻力位
        self.df = self.calculate_support_resistance(self.df, config['sr_period'])

        logger.info("技术指标计算完成")

        return self.df

    @staticmethod
    def _get_default_config() -> dict:
        """默认指标配置"""
        return {
            'ema_periods': [7, 25, 99],
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'kdj_period': 9,
            'kdj_smooth_k': 3,
            'kdj_smooth_d': 3,
            'williams_period': 14,
            'volume_ma_period': 20,
            'sr_period': 20
        }

    @staticmethod
    def calculate_ema(df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """
        计算指数移动平均线 (EMA)

        Args:
            df: 数据
            periods: 周期列表，如 [7, 25, 99]

        Returns:
            添加了EMA列的DataFrame
        """
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        计算MACD指标

        Returns:
            添加了macd, macd_signal, macd_histogram列的DataFrame
        """
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2
    ) -> pd.DataFrame:
        """
        计算布林带

        Returns:
            添加了bb_middle, bb_upper, bb_lower列的DataFrame
        """
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()

        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)

        # 计算布林带宽度（用于判断波动性）
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算相对强弱指标 (RSI)

        Returns:
            添加了rsi列的DataFrame
        """
        delta = df['close'].diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    @staticmethod
    def calculate_kdj(
        df: pd.DataFrame,
        period: int = 9,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> pd.DataFrame:
        """
        计算KDJ指标

        Returns:
            添加了kdj_k, kdj_d, kdj_j列的DataFrame
        """
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()

        rsv = (df['close'] - low_min) / (high_max - low_min) * 100

        df['kdj_k'] = rsv.ewm(com=smooth_k - 1, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=smooth_d - 1, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

        return df

    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算威廉指标 (Williams %R)

        Returns:
            添加了williams_r列的DataFrame
        """
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()

        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)

        return df

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算能量潮指标 (OBV)

        Returns:
            添加了obv列的DataFrame
        """
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        df['obv'] = obv

        # OBV移动平均
        df['obv_ma'] = df['obv'].rolling(window=20).mean()

        return df

    @staticmethod
    def calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算成交量移动平均

        Returns:
            添加了volume_ma列的DataFrame
        """
        df['volume_ma'] = df['volume'].rolling(window=period).mean()

        # 成交量相对比率
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        return df

    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算支撑位和阻力位

        使用滚动窗口内的最高价和最低价作为阻力位和支撑位

        Returns:
            添加了support, resistance列的DataFrame
        """
        df['resistance'] = df['high'].rolling(window=period).max()
        df['support'] = df['low'].rolling(window=period).min()

        # 计算距离支撑/阻力的百分比
        df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close'] * 100
        df['distance_to_support'] = (df['close'] - df['support']) / df['close'] * 100

        return df

    def get_trend_direction(self) -> pd.Series:
        """
        判断趋势方向

        Returns:
            Series: 1=上涨, 0=震荡, -1=下跌
        """
        conditions = [
            (self.df['ema_7'] > self.df['ema_25']) & (self.df['ema_25'] > self.df['ema_99']),
            (self.df['ema_7'] < self.df['ema_25']) & (self.df['ema_25'] < self.df['ema_99'])
        ]
        choices = [1, -1]

        return pd.Series(
            np.select(conditions, choices, default=0),
            index=self.df.index,
            name='trend'
        )

    def get_momentum_score(self) -> pd.Series:
        """
        计算综合动量得分 (-100 到 100)

        Returns:
            Series: 动量得分
        """
        # RSI得分 (-50 到 50)
        rsi_score = (self.df['rsi'] - 50)

        # KDJ得分 (-50 到 50)
        kdj_score = (self.df['kdj_k'] - 50)

        # Williams %R得分 (-50 到 50)
        williams_score = (self.df['williams_r'] + 50)

        # 综合得分
        momentum_score = (rsi_score + kdj_score + williams_score) / 3

        return momentum_score

    def get_volume_strength(self) -> pd.Series:
        """
        计算成交量强度

        Returns:
            Series: 0-100的成交量强度分数
        """
        # 基于volume_ratio计算
        volume_strength = self.df['volume_ratio'].clip(0, 3) * 100 / 3

        return volume_strength


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.rand(1000) * 1000
    })

    # 修正价格关系
    test_df['high'] = test_df[['open', 'high', 'close']].max(axis=1) + abs(np.random.randn(1000))
    test_df['low'] = test_df[['open', 'low', 'close']].min(axis=1) - abs(np.random.randn(1000))

    indicator = TechnicalIndicators(test_df)
    result_df = indicator.calculate_all_indicators()

    print("技术指标计算结果:")
    print(result_df.tail(10)[['close', 'ema_7', 'ema_25', 'rsi', 'macd', 'kdj_k']].to_string())
    print(f"\n总列数: {len(result_df.columns)}")
    print(f"列名: {result_df.columns.tolist()}")
