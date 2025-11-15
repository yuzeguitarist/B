"""
智能分析与决策引擎 - 基于技术指标生成交易信号
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum
import logging
from .funding_rate import FundingRateMonitor
from .long_short_ratio import LongShortRatioMonitor

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """信号强度"""
    STRONG = "强"
    MEDIUM = "中"
    WEAK = "弱"


class TrendType(Enum):
    """趋势类型"""
    UPTREND = "上涨"
    DOWNTREND = "下跌"
    SIDEWAYS = "震荡"


class TradingStrategy:
    """交易策略决策引擎"""

    def __init__(self, config: dict = None, symbol: str = "BTC/USDT", enable_live_data: bool = False):
        """
        初始化策略引擎

        Args:
            config: 策略配置参数
            symbol: 交易对符号
            enable_live_data: 是否启用实时数据（资金费率、多空比）
                             回测时应设为False，实盘时可设为True
        """
        self.config = config or self._get_default_config()
        self.symbol = symbol
        self.enable_live_data = enable_live_data

        # 初始化资金费率监控器（仅在启用实时数据时）
        if enable_live_data:
            self.funding_rate_monitor = FundingRateMonitor(self.config)
            self.ls_ratio_monitor = LongShortRatioMonitor(self.config)
            logger.info("实时数据已启用（资金费率、多空比）")
        else:
            self.funding_rate_monitor = None
            self.ls_ratio_monitor = None
            logger.info("实时数据已禁用（回测模式）")

    @staticmethod
    def _get_default_config() -> dict:
        """默认策略配置"""
        return {
            # RSI阈值
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_strong_oversold': 20,
            'rsi_strong_overbought': 80,

            # KDJ阈值
            'kdj_oversold': 20,
            'kdj_overbought': 80,

            # Williams %R阈值
            'williams_oversold': -80,
            'williams_overbought': -20,

            # 成交量阈值
            'volume_ratio_threshold': 1.5,  # 放量阈值

            # 布林带突破阈值
            'bb_breakout_threshold': 0.02,  # 2%

            # 综合得分阈值
            'buy_score_threshold': 60,
            'sell_score_threshold': -60,
            'strong_signal_threshold': 80,

            # 资金费率阈值
            'funding_rate_overbought': 0.001,  # 0.1%
            'funding_rate_oversold': -0.001,   # -0.1%
            'funding_rate_extreme': 0.002,     # 0.2%
            'funding_rate_weight': 15,         # 资金费率评分权重

            # 多空比阈值
            'extreme_long_ratio': 3.0,         # 多空比>3:1警告
            'extreme_short_ratio': 0.33,       # 多空比<1:3警告
            'crowded_threshold': 2.5,          # 拥挤阈值
            'long_short_ratio_weight': 10,     # 多空比评分权重
        }

    def analyze(self, df: pd.DataFrame, index: int) -> Dict:
        """
        分析当前时刻的市场状况并生成交易信号

        Args:
            df: 包含所有技术指标的DataFrame
            index: 当前分析的索引位置

        Returns:
            分析结果字典
        """
        if index < 100:  # 需要足够的历史数据
            return self._create_result(SignalType.HOLD, 0, "数据不足", {})

        row = df.iloc[index]
        prev_row = df.iloc[index - 1]

        # 1. 判断趋势
        trend = self._analyze_trend(df, index)

        # 2. 计算综合得分
        score_details = self._calculate_comprehensive_score(df, index)
        total_score = score_details['total_score']

        # 3. 生成信号
        signal_type, signal_strength, reasons = self._generate_signal(
            row, prev_row, total_score, trend, score_details
        )

        # 4. 风险评估
        risk_level = self._assess_risk(df, index, signal_type)

        return self._create_result(
            signal_type,
            total_score,
            reasons,
            score_details,
            trend,
            signal_strength,
            risk_level
        )

    def _analyze_trend(self, df: pd.DataFrame, index: int) -> TrendType:
        """
        分析趋势

        Args:
            df: 数据
            index: 当前索引

        Returns:
            趋势类型
        """
        row = df.iloc[index]

        # 基于EMA均线判断
        if row['ema_7'] > row['ema_25'] > row['ema_99']:
            # 短期、中期、长期均线多头排列
            return TrendType.UPTREND
        elif row['ema_7'] < row['ema_25'] < row['ema_99']:
            # 短期、中期、长期均线空头排列
            return TrendType.DOWNTREND
        else:
            # 均线交织，震荡行情
            return TrendType.SIDEWAYS

    def _calculate_comprehensive_score(self, df: pd.DataFrame, index: int) -> Dict:
        """
        计算综合得分（-100到100）

        Returns:
            得分详情字典
        """
        row = df.iloc[index]
        prev_row = df.iloc[index - 1]

        scores = {}

        # 1. RSI得分 (-20 到 20)
        rsi = row['rsi']
        if rsi < self.config['rsi_strong_oversold']:
            scores['rsi'] = 20
        elif rsi < self.config['rsi_oversold']:
            scores['rsi'] = 10
        elif rsi > self.config['rsi_strong_overbought']:
            scores['rsi'] = -20
        elif rsi > self.config['rsi_overbought']:
            scores['rsi'] = -10
        else:
            scores['rsi'] = (50 - rsi) / 5  # 线性映射

        # 2. KDJ得分 (-20 到 20)
        kdj_k = row['kdj_k']
        kdj_d = row['kdj_d']
        if kdj_k < self.config['kdj_oversold'] and kdj_k > kdj_d:
            scores['kdj'] = 20  # 超卖区金叉
        elif kdj_k < self.config['kdj_oversold']:
            scores['kdj'] = 10
        elif kdj_k > self.config['kdj_overbought'] and kdj_k < kdj_d:
            scores['kdj'] = -20  # 超买区死叉
        elif kdj_k > self.config['kdj_overbought']:
            scores['kdj'] = -10
        else:
            scores['kdj'] = (50 - kdj_k) / 5

        # 3. MACD得分 (-20 到 20)
        macd = row['macd']
        macd_signal = row['macd_signal']
        macd_hist = row['macd_histogram']
        prev_macd_hist = prev_row['macd_histogram']

        if macd_hist > 0 and prev_macd_hist <= 0:
            scores['macd'] = 20  # 金叉
        elif macd_hist < 0 and prev_macd_hist >= 0:
            scores['macd'] = -20  # 死叉
        elif macd_hist > 0:
            scores['macd'] = 10
        elif macd_hist < 0:
            scores['macd'] = -10
        else:
            scores['macd'] = 0

        # 4. 布林带得分 (-15 到 15)
        close = row['close']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        bb_middle = row['bb_middle']

        if close < bb_lower:
            scores['bollinger'] = 15  # 跌破下轨，超卖
        elif close > bb_upper:
            scores['bollinger'] = -15  # 突破上轨，超买
        elif close < bb_middle:
            scores['bollinger'] = 5
        elif close > bb_middle:
            scores['bollinger'] = -5
        else:
            scores['bollinger'] = 0

        # 5. 成交量得分 (-10 到 10)
        volume_ratio = row['volume_ratio']
        if volume_ratio > self.config['volume_ratio_threshold']:
            # 放量配合价格方向
            price_change = (close - prev_row['close']) / prev_row['close']
            if price_change > 0:
                scores['volume'] = 10  # 放量上涨
            else:
                scores['volume'] = -10  # 放量下跌
        else:
            scores['volume'] = 0

        # 6. 趋势得分 (-15 到 15)
        ema_7 = row['ema_7']
        ema_25 = row['ema_25']
        ema_99 = row['ema_99']

        if ema_7 > ema_25 > ema_99:
            scores['trend'] = 15  # 多头排列
        elif ema_7 < ema_25 < ema_99:
            scores['trend'] = -15  # 空头排列
        else:
            scores['trend'] = 0  # 震荡

        # 7. 资金费率得分 (-15 到 15)
        # 仅在启用实时数据时获取
        if self.enable_live_data and self.funding_rate_monitor:
            try:
                funding_score, funding_desc = self.funding_rate_monitor.get_funding_rate_score(self.symbol)
                scores['funding_rate'] = funding_score
                scores['funding_rate_desc'] = funding_desc
            except Exception as e:
                logger.warning(f"获取资金费率失败: {e}")
                scores['funding_rate'] = 0
                scores['funding_rate_desc'] = "资金费率获取失败"
        else:
            scores['funding_rate'] = 0
            scores['funding_rate_desc'] = "资金费率数据已禁用（回测模式）"

        # 8. 多空比得分 (-10 到 10)
        # 仅在启用实时数据时获取
        if self.enable_live_data and self.ls_ratio_monitor:
            try:
                ls_ratio_score, ls_ratio_desc = self.ls_ratio_monitor.get_ls_ratio_score(self.symbol)
                scores['long_short_ratio'] = ls_ratio_score
                scores['long_short_ratio_desc'] = ls_ratio_desc
            except Exception as e:
                logger.warning(f"获取多空比失败: {e}")
                scores['long_short_ratio'] = 0
                scores['long_short_ratio_desc'] = "多空比获取失败"
        else:
            scores['long_short_ratio'] = 0
            scores['long_short_ratio_desc'] = "多空比数据已禁用（回测模式）"

        # 计算总分（排除描述字段）
        total_score = sum(v for k, v in scores.items() if not k.endswith('_desc'))

        return {
            'total_score': total_score,
            'details': scores
        }

    def _generate_signal(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        total_score: float,
        trend: TrendType,
        score_details: Dict
    ) -> Tuple[SignalType, SignalStrength, str]:
        """
        生成交易信号

        Returns:
            (信号类型, 信号强度, 原因)
        """
        reasons = []

        # 判断信号类型
        if total_score >= self.config['buy_score_threshold']:
            signal_type = SignalType.BUY

            # 分析买入原因
            if score_details['details']['rsi'] > 10:
                reasons.append("RSI超卖反弹")
            if score_details['details']['kdj'] > 10:
                reasons.append("KDJ超卖区金叉")
            if score_details['details']['macd'] == 20:
                reasons.append("MACD金叉")
            if score_details['details']['bollinger'] > 10:
                reasons.append("跌破布林下轨")
            if score_details['details']['volume'] > 0:
                reasons.append("放量上涨")
            if trend == TrendType.UPTREND:
                reasons.append("趋势向上")
            if score_details['details'].get('funding_rate', 0) > 10:
                reasons.append("资金费率超卖（空头支付多头）")
            if score_details['details'].get('long_short_ratio', 0) > 5:
                reasons.append("多空比极低（空头拥挤）")

        elif total_score <= self.config['sell_score_threshold']:
            signal_type = SignalType.SELL

            # 分析卖出原因
            if score_details['details']['rsi'] < -10:
                reasons.append("RSI超买回调")
            if score_details['details']['kdj'] < -10:
                reasons.append("KDJ超买区死叉")
            if score_details['details']['macd'] == -20:
                reasons.append("MACD死叉")
            if score_details['details']['bollinger'] < -10:
                reasons.append("突破布林上轨")
            if score_details['details']['volume'] < 0:
                reasons.append("放量下跌")
            if trend == TrendType.DOWNTREND:
                reasons.append("趋势向下")
            if score_details['details'].get('funding_rate', 0) < -10:
                reasons.append("资金费率超买（多头支付空头）")
            if score_details['details'].get('long_short_ratio', 0) < -5:
                reasons.append("多空比极高（多头拥挤）")

        else:
            signal_type = SignalType.HOLD
            reasons.append("信号不明确，观望")

        # 判断信号强度
        if abs(total_score) >= self.config['strong_signal_threshold']:
            signal_strength = SignalStrength.STRONG
        elif abs(total_score) >= self.config['buy_score_threshold']:
            signal_strength = SignalStrength.MEDIUM
        else:
            signal_strength = SignalStrength.WEAK

        reason_str = ", ".join(reasons) if reasons else "无明确信号"

        return signal_type, signal_strength, reason_str

    def _assess_risk(
        self,
        df: pd.DataFrame,
        index: int,
        signal_type: SignalType
    ) -> str:
        """
        评估风险等级

        Returns:
            风险等级: 低/中/高
        """
        row = df.iloc[index]

        risk_factors = 0

        # 1. 检查布林带宽度（波动性）
        if row['bb_width'] > 0.05:  # 5%
            risk_factors += 1

        # 2. 检查距离支撑/阻力位
        if row['distance_to_resistance'] < 2:  # 距离阻力位<2%
            risk_factors += 1
        if row['distance_to_support'] < 2:  # 距离支撑位<2%
            risk_factors += 1

        # 3. 检查RSI极端值
        if row['rsi'] < 20 or row['rsi'] > 80:
            risk_factors += 1

        # 4. 检查近期价格波动
        recent_volatility = df.iloc[index-20:index]['close'].std() / row['close']
        if recent_volatility > 0.03:  # 3%
            risk_factors += 1

        # 根据风险因素数量评级
        if risk_factors >= 3:
            return "高"
        elif risk_factors >= 1:
            return "中"
        else:
            return "低"

    @staticmethod
    def _create_result(
        signal_type: SignalType,
        score: float,
        reasons: str,
        score_details: Dict,
        trend: TrendType = None,
        signal_strength: SignalStrength = None,
        risk_level: str = None
    ) -> Dict:
        """创建分析结果"""
        return {
            'signal': signal_type,
            'signal_strength': signal_strength,
            'score': score,
            'reasons': reasons,
            'score_details': score_details,
            'trend': trend,
            'risk_level': risk_level
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建测试数据（需要包含所有技术指标）
    from indicators import TechnicalIndicators

    dates = pd.date_range('2024-01-01', periods=500, freq='1min')
    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 105,
        'low': np.random.randn(500).cumsum() + 95,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.rand(500) * 1000
    })

    # 修正价格关系
    test_df['high'] = test_df[['open', 'high', 'close']].max(axis=1) + abs(np.random.randn(500))
    test_df['low'] = test_df[['open', 'low', 'close']].min(axis=1) - abs(np.random.randn(500))

    # 计算技术指标
    indicator = TechnicalIndicators(test_df)
    test_df = indicator.calculate_all_indicators()

    # 测试策略
    strategy = TradingStrategy()

    for i in range(200, 210):
        result = strategy.analyze(test_df, i)
        print(f"\n时间: {test_df.iloc[i]['timestamp']}")
        print(f"信号: {result['signal'].value}, 强度: {result['signal_strength'].value if result['signal_strength'] else 'N/A'}")
        print(f"得分: {result['score']:.2f}")
        print(f"趋势: {result['trend'].value if result['trend'] else 'N/A'}")
        print(f"原因: {result['reasons']}")
        print(f"风险: {result['risk_level']}")
