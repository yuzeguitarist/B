"""
波动率动态仓位管理模块 - 根据历史波动率动态调整仓位大小
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityPositionSizer:
    """基于波动率的动态仓位管理器"""

    def __init__(self, config: dict = None):
        """
        初始化波动率仓位管理器

        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()

        # 波动率计算参数
        self.volatility_period = self.config.get('volatility_period', 20)  # 波动率计算周期
        self.volatility_method = self.config.get('volatility_method', 'std')  # std或atr

        # 仓位计算参数
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 单次交易最大风险2%
        self.min_position_percent = self.config.get('min_position_percent', 0.1)  # 最小仓位10%
        self.max_position_percent = self.config.get('max_position_percent', 0.95)  # 最大仓位95%
        self.volatility_target = self.config.get('volatility_target', 0.02)  # 目标波动率2%

    @staticmethod
    def _get_default_config() -> dict:
        """默认配置"""
        return {
            'volatility_period': 20,           # 波动率计算周期（K线数量）
            'volatility_method': 'std',        # 波动率计算方法：std（标准差）或atr（真实波动幅度）
            'max_risk_per_trade': 0.02,        # 单次交易最大风险比例（2%）
            'min_position_percent': 0.1,       # 最小仓位比例（10%）
            'max_position_percent': 0.95,      # 最大仓位比例（95%）
            'volatility_target': 0.02,         # 目标波动率（2%）
            'use_volatility_position': True,   # 是否启用波动率仓位管理
        }

    def calculate_volatility(self, df: pd.DataFrame, index: int) -> Optional[float]:
        """
        计算历史波动率

        Args:
            df: 包含价格数据的DataFrame
            index: 当前索引位置

        Returns:
            波动率值（小数形式）
        """
        if index < self.volatility_period:
            logger.warning(f"数据不足，无法计算波动率（需要至少{self.volatility_period}根K线）")
            return None

        # 获取历史数据
        history = df.iloc[index - self.volatility_period:index]

        if self.volatility_method == 'std':
            # 使用收益率标准差计算波动率
            returns = history['close'].pct_change().dropna()
            volatility = returns.std()

        elif self.volatility_method == 'atr':
            # 使用ATR（真实波动幅度）计算波动率
            if 'atr' in history.columns:
                # 如果已经计算了ATR，直接使用
                volatility = history['atr'].iloc[-1] / history['close'].iloc[-1]
            else:
                # 手动计算ATR
                high_low = history['high'] - history['low']
                high_close = np.abs(history['high'] - history['close'].shift())
                low_close = np.abs(history['low'] - history['close'].shift())

                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean().iloc[-1]
                volatility = atr / history['close'].iloc[-1]

        else:
            logger.error(f"不支持的波动率计算方法: {self.volatility_method}")
            return None

        return volatility

    def calculate_position_size(
        self,
        current_capital: float,
        current_price: float,
        volatility: float,
        signal_strength: str = "中"
    ) -> Tuple[float, float, str]:
        """
        基于波动率计算仓位大小

        Args:
            current_capital: 当前资金
            current_price: 当前价格
            volatility: 当前波动率
            signal_strength: 信号强度（强/中/弱）

        Returns:
            (仓位比例, 买入金额, 说明)
        """
        if volatility is None or volatility <= 0:
            # 如果无法获取波动率，使用默认仓位
            logger.warning("波动率无效，使用默认仓位")
            default_position = 0.5
            return default_position, current_capital * default_position, "波动率数据不可用，使用默认仓位"

        # 方法1: 基于固定风险的仓位计算
        # 仓位 = (可接受亏损 / 波动率)
        # 可接受亏损 = 总资金 × 单次交易风险比例
        acceptable_loss = current_capital * self.max_risk_per_trade
        position_value = acceptable_loss / volatility

        # 转换为仓位比例
        position_percent = position_value / current_capital

        # 方法2: 基于目标波动率的仓位计算
        # 仓位 = 目标波动率 / 实际波动率
        target_position = self.volatility_target / volatility if volatility > 0 else 1.0

        # 取两种方法的平均值
        position_percent = (position_percent + target_position) / 2

        # 根据信号强度调整仓位
        strength_multiplier = {
            '强': 1.2,
            '中': 1.0,
            '弱': 0.7
        }.get(signal_strength, 1.0)

        position_percent *= strength_multiplier

        # 限制仓位在合理范围内
        position_percent = max(self.min_position_percent, min(position_percent, self.max_position_percent))

        # 计算买入金额
        buy_amount = current_capital * position_percent

        # 生成说明
        description = (
            f"波动率: {volatility*100:.2f}%, "
            f"目标波动率: {self.volatility_target*100:.2f}%, "
            f"信号强度: {signal_strength}, "
            f"建议仓位: {position_percent*100:.1f}%"
        )

        logger.info(description)

        return position_percent, buy_amount, description

    def get_volatility_adjusted_stop_loss(
        self,
        entry_price: float,
        volatility: float,
        multiplier: float = 2.0
    ) -> float:
        """
        基于波动率计算动态止损价格

        Args:
            entry_price: 入场价格
            volatility: 波动率
            multiplier: 波动率倍数（默认2倍）

        Returns:
            止损价格
        """
        if volatility is None or volatility <= 0:
            # 使用固定2%止损
            return entry_price * 0.98

        # 止损距离 = 价格 × 波动率 × 倍数
        stop_distance = entry_price * volatility * multiplier

        # 止损价格 = 入场价格 - 止损距离
        stop_loss_price = entry_price - stop_distance

        return stop_loss_price

    def get_volatility_adjusted_take_profit(
        self,
        entry_price: float,
        volatility: float,
        multiplier: float = 3.0
    ) -> float:
        """
        基于波动率计算动态止盈价格

        Args:
            entry_price: 入场价格
            volatility: 波动率
            multiplier: 波动率倍数（默认3倍）

        Returns:
            止盈价格
        """
        if volatility is None or volatility <= 0:
            # 使用固定5%止盈
            return entry_price * 1.05

        # 止盈距离 = 价格 × 波动率 × 倍数
        profit_distance = entry_price * volatility * multiplier

        # 止盈价格 = 入场价格 + 止盈距离
        take_profit_price = entry_price + profit_distance

        return take_profit_price

    def analyze_volatility_regime(self, volatility: float) -> str:
        """
        分析波动率状态

        Args:
            volatility: 当前波动率

        Returns:
            波动率状态描述
        """
        if volatility is None:
            return "未知"

        if volatility < 0.01:
            return "极低波动"
        elif volatility < 0.02:
            return "低波动"
        elif volatility < 0.03:
            return "正常波动"
        elif volatility < 0.05:
            return "高波动"
        else:
            return "极高波动"


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    np.random.seed(42)

    # 模拟价格数据（带波动）
    returns = np.random.randn(100) * 0.02  # 2%波动率
    prices = 50000 * (1 + returns).cumprod()

    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.rand(100) * 1000
    })

    # 创建波动率仓位管理器
    position_sizer = VolatilityPositionSizer()

    # 测试波动率计算
    print("===== 波动率计算测试 =====")
    for i in [30, 50, 70, 90]:
        volatility = position_sizer.calculate_volatility(test_df, i)
        if volatility:
            print(f"第{i}根K线 - 波动率: {volatility*100:.2f}% ({position_sizer.analyze_volatility_regime(volatility)})")

    # 测试仓位计算
    print("\n===== 仓位计算测试 =====")
    current_capital = 10000
    current_price = test_df.iloc[50]['close']
    volatility = position_sizer.calculate_volatility(test_df, 50)

    for strength in ['强', '中', '弱']:
        pos_pct, buy_amt, desc = position_sizer.calculate_position_size(
            current_capital, current_price, volatility, strength
        )
        print(f"\n信号强度: {strength}")
        print(f"  仓位比例: {pos_pct*100:.1f}%")
        print(f"  买入金额: ${buy_amt:.2f}")
        print(f"  说明: {desc}")

    # 测试动态止损止盈
    print("\n===== 动态止损止盈测试 =====")
    entry_price = current_price
    stop_loss = position_sizer.get_volatility_adjusted_stop_loss(entry_price, volatility)
    take_profit = position_sizer.get_volatility_adjusted_take_profit(entry_price, volatility)

    print(f"入场价格: ${entry_price:.2f}")
    print(f"动态止损: ${stop_loss:.2f} ({(stop_loss/entry_price-1)*100:.2f}%)")
    print(f"动态止盈: ${take_profit:.2f} ({(take_profit/entry_price-1)*100:.2f}%)")
