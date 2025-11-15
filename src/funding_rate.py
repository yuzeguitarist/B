"""
资金费率监控模块 - 监控永续合约资金费率以判断市场情绪
"""
import ccxt
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class FundingRateMonitor:
    """资金费率监控器"""

    def __init__(self, config: dict = None):
        """
        初始化资金费率监控器

        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()

        # 初始化交易所（币安期货）
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 使用期货市场
                }
            })
        except Exception as e:
            logger.error(f"初始化币安期货交易所失败: {e}")
            self.exchange = None

        # 资金费率阈值
        self.overbought_threshold = self.config.get('funding_rate_overbought', 0.001)  # 0.1%
        self.oversold_threshold = self.config.get('funding_rate_oversold', -0.001)  # -0.1%
        self.extreme_threshold = self.config.get('funding_rate_extreme', 0.002)  # 0.2%

        # 缓存（按symbol分别缓存）
        self.cache = {}  # {symbol: {'rate': float, 'time': datetime}}
        self.cache_duration = 300  # 缓存5分钟

    @staticmethod
    def _get_default_config() -> dict:
        """默认配置"""
        return {
            'funding_rate_overbought': 0.001,  # 0.1% - 超买阈值
            'funding_rate_oversold': -0.001,   # -0.1% - 超卖阈值
            'funding_rate_extreme': 0.002,     # 0.2% - 极端阈值
            'funding_rate_weight': 15,         # 在策略评分中的权重
        }

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        获取指定交易对的当前资金费率

        Args:
            symbol: 交易对符号，如 'BTC/USDT'

        Returns:
            资金费率（小数形式），如 0.0001 表示 0.01%
            如果获取失败返回 None
        """
        if not self.exchange:
            logger.warning("交易所未初始化，无法获取资金费率")
            return None

        # 检查缓存
        if self._use_cache(symbol):
            cached_rate = self.cache[symbol]['rate']
            logger.debug(f"使用缓存的{symbol}资金费率: {cached_rate}")
            return cached_rate

        try:
            # 转换为期货交易对格式
            futures_symbol = symbol.replace('/', '')  # BTC/USDT -> BTCUSDT

            # 获取资金费率
            # 币安API返回的funding_rate是实际的费率值（已经是小数形式）
            funding_info = self.exchange.fapiPublic_get_premiumindex({
                'symbol': futures_symbol
            })

            funding_rate = float(funding_info['lastFundingRate'])

            # 更新缓存（按symbol存储）
            self.cache[symbol] = {
                'rate': funding_rate,
                'time': datetime.now()
            }

            logger.info(f"{symbol} 当前资金费率: {funding_rate:.6f} ({funding_rate*100:.4f}%)")

            return funding_rate

        except Exception as e:
            logger.error(f"获取资金费率失败: {e}")
            return None

    def _use_cache(self, symbol: str) -> bool:
        """检查是否使用缓存数据"""
        if symbol not in self.cache:
            return False

        elapsed = (datetime.now() - self.cache[symbol]['time']).total_seconds()
        return elapsed < self.cache_duration

    def analyze_funding_rate(self, funding_rate: float) -> Dict:
        """
        分析资金费率并返回市场情绪

        Args:
            funding_rate: 资金费率

        Returns:
            分析结果字典，包含：
            - sentiment: 情绪（超买/中性/超卖）
            - signal: 交易信号（看空/观望/看多）
            - score: 评分（-20到20）
            - warning: 警告信息
        """
        if funding_rate is None:
            return {
                'sentiment': '未知',
                'signal': '观望',
                'score': 0,
                'warning': '无法获取资金费率数据',
                'funding_rate': None,
                'funding_rate_percent': None
            }

        # 判断情绪和信号
        if funding_rate > self.extreme_threshold:
            sentiment = '极度超买'
            signal = '强看空'
            score = -20  # 负分表示看空
            warning = f'资金费率极高 ({funding_rate*100:.4f}%)，多头支付空头，市场可能过热，警惕回调风险！'
        elif funding_rate > self.overbought_threshold:
            sentiment = '超买'
            signal = '看空'
            score = -15
            warning = f'资金费率偏高 ({funding_rate*100:.4f}%)，多头情绪高涨，注意风险'
        elif funding_rate < -self.extreme_threshold:
            sentiment = '极度超卖'
            signal = '强看多'
            score = 20  # 正分表示看多
            warning = f'资金费率极低 ({funding_rate*100:.4f}%)，空头支付多头，市场可能超跌，关注反弹机会！'
        elif funding_rate < self.oversold_threshold:
            sentiment = '超卖'
            signal = '看多'
            score = 15
            warning = f'资金费率偏低 ({funding_rate*100:.4f}%)，空头情绪高涨，可能反弹'
        else:
            # 中性区域，线性映射到 -10 到 10
            sentiment = '中性'
            signal = '观望'
            # 在 -0.001 到 0.001 之间线性映射
            score = (funding_rate / 0.001) * 10
            score = max(-10, min(10, score))  # 限制在 -10 到 10
            warning = None

        return {
            'sentiment': sentiment,
            'signal': signal,
            'score': score,
            'warning': warning,
            'funding_rate': funding_rate,
            'funding_rate_percent': funding_rate * 100
        }

    def get_funding_rate_score(self, symbol: str) -> Tuple[float, str]:
        """
        获取资金费率对应的策略评分

        Args:
            symbol: 交易对符号

        Returns:
            (评分, 说明文本)
        """
        funding_rate = self.get_funding_rate(symbol)
        analysis = self.analyze_funding_rate(funding_rate)

        # 应用权重（注意：权重已经在默认配置中定义为15，这里直接使用analysis的score）
        # analysis['score']的范围已经是 -20到20，这里返回时不需要再乘以权重
        # 因为在策略评分系统中已经设计好了各个指标的权重
        score = analysis['score']

        # 生成说明文本
        if analysis['warning']:
            description = analysis['warning']
        else:
            if funding_rate is not None:
                description = f"资金费率正常 ({funding_rate*100:.4f}%)"
            else:
                description = "无资金费率数据"

        return score, description

    def get_next_funding_time(self, symbol: str) -> Optional[datetime]:
        """
        获取下一次资金费率结算时间

        Args:
            symbol: 交易对符号

        Returns:
            下一次结算时间
        """
        if not self.exchange:
            return None

        try:
            futures_symbol = symbol.replace('/', '')
            funding_info = self.exchange.fapiPublic_get_premiumindex({
                'symbol': futures_symbol
            })

            # nextFundingTime 是毫秒时间戳
            next_funding_timestamp = int(funding_info['nextFundingTime']) / 1000
            next_funding_time = datetime.fromtimestamp(next_funding_timestamp)

            return next_funding_time

        except Exception as e:
            logger.error(f"获取下一次资金费率结算时间失败: {e}")
            return None

    def get_funding_rate_history(self, symbol: str, limit: int = 100) -> list:
        """
        获取历史资金费率

        Args:
            symbol: 交易对符号
            limit: 获取条数

        Returns:
            历史资金费率列表
        """
        if not self.exchange:
            return []

        try:
            futures_symbol = symbol.replace('/', '')

            # 获取历史资金费率
            history = self.exchange.fapiPublic_get_fundingrate({
                'symbol': futures_symbol,
                'limit': limit
            })

            # 解析数据
            funding_history = []
            for item in history:
                funding_history.append({
                    'timestamp': datetime.fromtimestamp(int(item['fundingTime']) / 1000),
                    'funding_rate': float(item['fundingRate']),
                    'funding_rate_percent': float(item['fundingRate']) * 100
                })

            return funding_history

        except Exception as e:
            logger.error(f"获取历史资金费率失败: {e}")
            return []


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建监控器
    monitor = FundingRateMonitor()

    # 测试BTC/USDT的资金费率
    symbol = 'BTC/USDT'

    print(f"\n===== {symbol} 资金费率分析 =====")

    # 获取当前资金费率
    funding_rate = monitor.get_funding_rate(symbol)
    if funding_rate is not None:
        print(f"当前资金费率: {funding_rate:.6f} ({funding_rate*100:.4f}%)")

        # 分析资金费率
        analysis = monitor.analyze_funding_rate(funding_rate)
        print(f"市场情绪: {analysis['sentiment']}")
        print(f"交易信号: {analysis['signal']}")
        print(f"策略评分: {analysis['score']:.2f}")
        if analysis['warning']:
            print(f"⚠️  警告: {analysis['warning']}")

        # 获取下一次结算时间
        next_time = monitor.get_next_funding_time(symbol)
        if next_time:
            print(f"下次结算: {next_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 获取历史资金费率（最近10次）
        print("\n最近10次资金费率:")
        history = monitor.get_funding_rate_history(symbol, limit=10)
        for item in history:
            print(f"{item['timestamp'].strftime('%Y-%m-%d %H:%M')} - {item['funding_rate']:.6f} ({item['funding_rate_percent']:.4f}%)")
    else:
        print("无法获取资金费率数据")

    # 测试评分功能
    print("\n===== 策略评分测试 =====")
    score, description = monitor.get_funding_rate_score(symbol)
    print(f"评分: {score:.2f}")
    print(f"说明: {description}")
