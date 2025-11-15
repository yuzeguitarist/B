"""
多空比监控模块 - 监控市场多空比例和持仓量以判断市场情绪
"""
import ccxt
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class LongShortRatioMonitor:
    """多空比监控器"""

    def __init__(self, config: dict = None):
        """
        初始化多空比监控器

        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()

        # 初始化交易所（币安期货）
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
        except Exception as e:
            logger.error(f"初始化币安期货交易所失败: {e}")
            self.exchange = None

        # 多空比阈值
        self.extreme_long_ratio = self.config.get('extreme_long_ratio', 3.0)  # 多空比>3:1警告
        self.extreme_short_ratio = self.config.get('extreme_short_ratio', 0.33)  # 多空比<1:3警告
        self.crowded_threshold = self.config.get('crowded_threshold', 2.5)  # 拥挤阈值

        # 缓存
        self.last_ratio = None
        self.last_oi = None
        self.last_update_time = None
        self.cache_duration = 300  # 缓存5分钟

    @staticmethod
    def _get_default_config() -> dict:
        """默认配置"""
        return {
            'extreme_long_ratio': 3.0,       # 多空比>3:1警告超买
            'extreme_short_ratio': 0.33,     # 多空比<1:3警告超卖
            'crowded_threshold': 2.5,        # 拥挤阈值
            'long_short_ratio_weight': 10,   # 在策略评分中的权重
        }

    def get_long_short_ratio(self, symbol: str, period: str = '5m') -> Optional[float]:
        """
        获取账户多空比（基于账户数量）

        Args:
            symbol: 交易对符号，如 'BTC/USDT'
            period: 时间周期，如 '5m', '15m', '1h'

        Returns:
            多空比值（多头账户数/空头账户数）
        """
        if not self.exchange:
            logger.warning("交易所未初始化，无法获取多空比")
            return None

        # 检查缓存
        if self._use_cache():
            logger.debug(f"使用缓存的多空比: {self.last_ratio}")
            return self.last_ratio

        try:
            # 转换为期货交易对格式
            futures_symbol = symbol.replace('/', '')

            # 获取全局多空比（基于账户数量）
            # 币安API: GET /futures/data/globalLongShortAccountRatio
            ratio_data = self.exchange.fapiData_get_globalLongShortAccountRatio({
                'symbol': futures_symbol,
                'period': period,
                'limit': 1
            })

            if ratio_data and len(ratio_data) > 0:
                latest_data = ratio_data[0]
                long_ratio = float(latest_data['longAccount'])
                short_ratio = float(latest_data['shortAccount'])

                # 计算多空比
                if short_ratio > 0:
                    ls_ratio = long_ratio / short_ratio
                else:
                    ls_ratio = 999.0  # 如果空头为0，设为极大值

                # 更新缓存
                self.last_ratio = ls_ratio
                self.last_update_time = datetime.now()

                logger.info(
                    f"{symbol} 多空比: {ls_ratio:.2f} "
                    f"(多头: {long_ratio*100:.1f}%, 空头: {short_ratio*100:.1f}%)"
                )

                return ls_ratio

        except Exception as e:
            logger.error(f"获取多空比失败: {e}")
            return None

    def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """
        获取持仓量（Open Interest）

        Args:
            symbol: 交易对符号

        Returns:
            持仓量信息字典
        """
        if not self.exchange:
            logger.warning("交易所未初始化，无法获取持仓量")
            return None

        try:
            futures_symbol = symbol.replace('/', '')

            # 获取持仓量
            oi_data = self.exchange.fapiPublic_get_openinterest({
                'symbol': futures_symbol
            })

            open_interest = float(oi_data['openInterest'])
            timestamp = int(oi_data['time'])

            # 更新缓存
            self.last_oi = {
                'open_interest': open_interest,
                'timestamp': datetime.fromtimestamp(timestamp / 1000),
                'symbol': symbol
            }

            logger.info(f"{symbol} 持仓量: {open_interest:.2f}")

            return self.last_oi

        except Exception as e:
            logger.error(f"获取持仓量失败: {e}")
            return None

    def _use_cache(self) -> bool:
        """检查是否使用缓存数据"""
        if self.last_ratio is None or self.last_update_time is None:
            return False

        elapsed = (datetime.now() - self.last_update_time).total_seconds()
        return elapsed < self.cache_duration

    def analyze_long_short_ratio(self, ls_ratio: float) -> Dict:
        """
        分析多空比并返回市场情绪

        Args:
            ls_ratio: 多空比值

        Returns:
            分析结果字典
        """
        if ls_ratio is None:
            return {
                'sentiment': '未知',
                'signal': '观望',
                'score': 0,
                'warning': '无法获取多空比数据',
                'ls_ratio': None,
                'crowded': False
            }

        # 判断市场情绪
        if ls_ratio > self.extreme_long_ratio:
            sentiment = '极度看多'
            signal = '看空（逆向）'
            score = -10  # 多头过度拥挤，看空
            warning = f'多空比极高 ({ls_ratio:.2f}:1)，多头严重拥挤，警惕反转风险！'
            crowded = True

        elif ls_ratio > self.crowded_threshold:
            sentiment = '偏向看多'
            signal = '谨慎看空'
            score = -5
            warning = f'多空比偏高 ({ls_ratio:.2f}:1)，多头拥挤，注意风险'
            crowded = True

        elif ls_ratio < self.extreme_short_ratio:
            sentiment = '极度看空'
            signal = '看多（逆向）'
            score = 10  # 空头过度拥挤，看多
            warning = f'多空比极低 ({ls_ratio:.2f}:1)，空头严重拥挤，关注反弹机会！'
            crowded = True

        elif ls_ratio < (1 / self.crowded_threshold):
            sentiment = '偏向看空'
            signal = '谨慎看多'
            score = 5
            warning = f'多空比偏低 ({ls_ratio:.2f}:1)，空头拥挤，可能反弹'
            crowded = True

        else:
            # 中性区域
            sentiment = '均衡'
            signal = '观望'
            # 在 0.5 到 2.0 之间线性映射到 -3 到 3
            if ls_ratio > 1.0:
                score = min(3, (ls_ratio - 1.0) / 0.5 * -3)
            else:
                score = max(-3, (1.0 - ls_ratio) / 0.5 * 3)
            warning = None
            crowded = False

        return {
            'sentiment': sentiment,
            'signal': signal,
            'score': score,
            'warning': warning,
            'ls_ratio': ls_ratio,
            'crowded': crowded
        }

    def get_ls_ratio_score(self, symbol: str, period: str = '5m') -> Tuple[float, str]:
        """
        获取多空比对应的策略评分

        Args:
            symbol: 交易对符号
            period: 时间周期

        Returns:
            (评分, 说明文本)
        """
        ls_ratio = self.get_long_short_ratio(symbol, period)
        analysis = self.analyze_long_short_ratio(ls_ratio)

        # 生成说明文本
        if analysis['warning']:
            description = analysis['warning']
        else:
            if ls_ratio is not None:
                description = f"多空比正常 ({ls_ratio:.2f}:1)"
            else:
                description = "无多空比数据"

        return analysis['score'], description

    def get_top_positions_ratio(self, symbol: str, period: str = '5m') -> Optional[Dict]:
        """
        获取大户多空比（基于持仓量）

        Args:
            symbol: 交易对符号
            period: 时间周期

        Returns:
            大户多空比信息
        """
        if not self.exchange:
            return None

        try:
            futures_symbol = symbol.replace('/', '')

            # 获取大户多空比（基于持仓量）
            top_ratio_data = self.exchange.fapiData_get_topLongShortPositionRatio({
                'symbol': futures_symbol,
                'period': period,
                'limit': 1
            })

            if top_ratio_data and len(top_ratio_data) > 0:
                latest = top_ratio_data[0]
                long_position = float(latest['longAccount'])
                short_position = float(latest['shortAccount'])

                if short_position > 0:
                    top_ls_ratio = long_position / short_position
                else:
                    top_ls_ratio = 999.0

                return {
                    'top_ls_ratio': top_ls_ratio,
                    'long_position_percent': long_position * 100,
                    'short_position_percent': short_position * 100,
                    'timestamp': datetime.fromtimestamp(int(latest['timestamp']) / 1000)
                }

        except Exception as e:
            logger.error(f"获取大户多空比失败: {e}")
            return None

    def analyze_market_sentiment(self, symbol: str) -> Dict:
        """
        综合分析市场情绪（结合多空比、持仓量等）

        Args:
            symbol: 交易对符号

        Returns:
            综合分析结果
        """
        # 获取账户多空比
        account_ratio = self.get_long_short_ratio(symbol)
        account_analysis = self.analyze_long_short_ratio(account_ratio)

        # 获取大户多空比
        top_ratio_data = self.get_top_positions_ratio(symbol)

        # 获取持仓量
        oi_data = self.get_open_interest(symbol)

        # 综合评分
        total_score = account_analysis['score']

        # 如果大户多空比与散户一致，增加权重
        if top_ratio_data:
            top_ratio = top_ratio_data['top_ls_ratio']
            top_analysis = self.analyze_long_short_ratio(top_ratio)

            # 如果大户和散户方向一致，增强信号
            if (account_analysis['score'] > 0 and top_analysis['score'] > 0) or \
               (account_analysis['score'] < 0 and top_analysis['score'] < 0):
                total_score *= 1.5  # 增强50%

        return {
            'account_ratio': account_ratio,
            'account_sentiment': account_analysis['sentiment'],
            'account_score': account_analysis['score'],
            'top_ratio_data': top_ratio_data,
            'open_interest': oi_data,
            'total_score': total_score,
            'warning': account_analysis['warning'],
            'crowded': account_analysis['crowded']
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建监控器
    monitor = LongShortRatioMonitor()

    symbol = 'BTC/USDT'

    print(f"\n===== {symbol} 多空比分析 =====\n")

    # 获取多空比
    ls_ratio = monitor.get_long_short_ratio(symbol)
    if ls_ratio:
        print(f"账户多空比: {ls_ratio:.2f}:1")

        # 分析多空比
        analysis = monitor.analyze_long_short_ratio(ls_ratio)
        print(f"市场情绪: {analysis['sentiment']}")
        print(f"交易信号: {analysis['signal']}")
        print(f"策略评分: {analysis['score']:.2f}")
        print(f"是否拥挤: {'是' if analysis['crowded'] else '否'}")
        if analysis['warning']:
            print(f"⚠️  警告: {analysis['warning']}")

    # 获取大户多空比
    print("\n===== 大户多空比 =====")
    top_ratio = monitor.get_top_positions_ratio(symbol)
    if top_ratio:
        print(f"大户多空比: {top_ratio['top_ls_ratio']:.2f}:1")
        print(f"大户多头占比: {top_ratio['long_position_percent']:.1f}%")
        print(f"大户空头占比: {top_ratio['short_position_percent']:.1f}%")

    # 获取持仓量
    print("\n===== 持仓量 =====")
    oi = monitor.get_open_interest(symbol)
    if oi:
        print(f"持仓量: {oi['open_interest']:.2f}")
        print(f"更新时间: {oi['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # 综合分析
    print("\n===== 综合情绪分析 =====")
    sentiment = monitor.analyze_market_sentiment(symbol)
    print(f"账户多空比: {sentiment['account_ratio']:.2f}:1 ({sentiment['account_sentiment']})")
    if sentiment['top_ratio_data']:
        print(f"大户多空比: {sentiment['top_ratio_data']['top_ls_ratio']:.2f}:1")
    print(f"综合评分: {sentiment['total_score']:.2f}")
    print(f"市场拥挤: {'是' if sentiment['crowded'] else '否'}")
    if sentiment['warning']:
        print(f"⚠️  {sentiment['warning']}")

    # 测试评分功能
    print("\n===== 策略评分测试 =====")
    score, description = monitor.get_ls_ratio_score(symbol)
    print(f"评分: {score:.2f}")
    print(f"说明: {description}")
