"""
移动止损模块 - 盈利后自动上移止损价格保护利润
"""
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class TrailingStopManager:
    """移动止损管理器"""

    def __init__(self, config: dict = None):
        """
        初始化移动止损管理器

        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()

        # 移动止损参数
        self.trailing_stop_activation = self.config.get('trailing_stop_activation', 0.02)  # 盈利2%后激活
        self.trailing_stop_distance = self.config.get('trailing_stop_distance', 0.03)  # 回撤3%触发
        self.trailing_stop_step = self.config.get('trailing_stop_step', 0.01)  # 止损价上移步长1%

        # 状态跟踪
        self.highest_price = None  # 最高价格
        self.trailing_stop_price = None  # 移动止损价格
        self.is_activated = False  # 是否已激活移动止损

    @staticmethod
    def _get_default_config() -> dict:
        """默认配置"""
        return {
            'use_trailing_stop': True,          # 是否启用移动止损
            'trailing_stop_activation': 0.02,   # 盈利多少后激活移动止损（2%）
            'trailing_stop_distance': 0.03,     # 从最高点回撤多少触发止损（3%）
            'trailing_stop_step': 0.01,         # 止损价上移的最小步长（1%）
        }

    def initialize(self, entry_price: float, initial_stop_loss: float):
        """
        初始化移动止损（开仓时调用）

        Args:
            entry_price: 入场价格
            initial_stop_loss: 初始止损价格
        """
        self.highest_price = entry_price
        self.trailing_stop_price = initial_stop_loss
        self.is_activated = False

        logger.info(f"移动止损初始化 - 入场价: {entry_price:.2f}, 初始止损: {initial_stop_loss:.2f}")

    def update(self, current_price: float, entry_price: float) -> Tuple[float, bool, str]:
        """
        更新移动止损价格

        Args:
            current_price: 当前价格
            entry_price: 入场价格

        Returns:
            (新的止损价格, 是否触发止损, 说明信息)
        """
        if self.trailing_stop_price is None:
            logger.error("移动止损未初始化，请先调用initialize()")
            return entry_price * 0.98, False, "移动止损未初始化"

        # 更新最高价格
        if current_price > self.highest_price:
            old_highest = self.highest_price
            self.highest_price = current_price

            # 检查是否达到激活条件
            profit_percent = (current_price - entry_price) / entry_price
            if not self.is_activated and profit_percent >= self.trailing_stop_activation:
                self.is_activated = True
                logger.info(f"移动止损已激活 - 当前盈利: {profit_percent*100:.2f}%")

            # 如果已激活，更新止损价格
            if self.is_activated:
                # 新的止损价 = 最高价 × (1 - 回撤比例)
                new_stop = self.highest_price * (1 - self.trailing_stop_distance)

                # 止损价只能上移，不能下移
                if new_stop > self.trailing_stop_price:
                    # 检查是否满足最小步长要求
                    price_increase = (new_stop - self.trailing_stop_price) / self.trailing_stop_price
                    if price_increase >= self.trailing_stop_step:
                        old_stop = self.trailing_stop_price
                        self.trailing_stop_price = new_stop
                        logger.info(
                            f"止损价上移 - 最高价: {old_highest:.2f} -> {self.highest_price:.2f}, "
                            f"止损价: {old_stop:.2f} -> {self.trailing_stop_price:.2f}"
                        )

        # 检查是否触发止损
        triggered = False
        message = ""

        if current_price <= self.trailing_stop_price:
            triggered = True
            profit_percent = (current_price - entry_price) / entry_price
            drawdown_from_high = (self.highest_price - current_price) / self.highest_price

            message = (
                f"触发移动止损 - "
                f"最高价: {self.highest_price:.2f}, "
                f"当前价: {current_price:.2f}, "
                f"止损价: {self.trailing_stop_price:.2f}, "
                f"回撤: {drawdown_from_high*100:.2f}%, "
                f"总盈亏: {profit_percent*100:.2f}%"
            )
            logger.warning(message)

        return self.trailing_stop_price, triggered, message

    def get_status(self, current_price: float, entry_price: float) -> Dict:
        """
        获取移动止损状态信息

        Args:
            current_price: 当前价格
            entry_price: 入场价格

        Returns:
            状态信息字典
        """
        if self.trailing_stop_price is None:
            return {
                'is_initialized': False,
                'is_activated': False,
                'message': '移动止损未初始化'
            }

        profit_percent = (current_price - entry_price) / entry_price
        distance_to_stop = (current_price - self.trailing_stop_price) / current_price
        distance_from_high = (self.highest_price - current_price) / self.highest_price

        return {
            'is_initialized': True,
            'is_activated': self.is_activated,
            'entry_price': entry_price,
            'current_price': current_price,
            'highest_price': self.highest_price,
            'trailing_stop_price': self.trailing_stop_price,
            'profit_percent': profit_percent * 100,
            'distance_to_stop_percent': distance_to_stop * 100,
            'distance_from_high_percent': distance_from_high * 100,
            'activation_threshold': self.trailing_stop_activation * 100,
            'trailing_distance': self.trailing_stop_distance * 100,
        }

    def reset(self):
        """重置移动止损状态"""
        self.highest_price = None
        self.trailing_stop_price = None
        self.is_activated = False
        logger.info("移动止损已重置")


class MultiLevelTrailingStop:
    """多级移动止损管理器（更激进的利润保护）"""

    def __init__(self, config: dict = None):
        """
        初始化多级移动止损管理器

        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()

        # 定义多个止损级别
        # 格式：(盈利阈值, 回撤比例)
        self.stop_levels = [
            (0.02, 0.03),  # 盈利2%后，回撤3%触发
            (0.05, 0.025), # 盈利5%后，回撤2.5%触发
            (0.10, 0.02),  # 盈利10%后，回撤2%触发
            (0.20, 0.015), # 盈利20%后，回撤1.5%触发
        ]

        # 状态
        self.current_level = -1  # 当前级别
        self.highest_price = None
        self.trailing_stop_price = None

    @staticmethod
    def _get_default_config() -> dict:
        """默认配置"""
        return {
            'use_multilevel_trailing_stop': False,  # 是否使用多级移动止损
        }

    def initialize(self, entry_price: float, initial_stop_loss: float):
        """初始化多级移动止损"""
        self.highest_price = entry_price
        self.trailing_stop_price = initial_stop_loss
        self.current_level = -1
        logger.info(f"多级移动止损初始化 - 入场价: {entry_price:.2f}")

    def update(self, current_price: float, entry_price: float) -> Tuple[float, bool, str]:
        """
        更新多级移动止损

        Args:
            current_price: 当前价格
            entry_price: 入场价格

        Returns:
            (止损价格, 是否触发, 说明)
        """
        # 检查是否已初始化
        if self.highest_price is None or self.trailing_stop_price is None:
            logger.error("多级移动止损未初始化，请先调用initialize()")
            return entry_price * 0.98, False, "多级移动止损未初始化"

        # 更新最高价
        if current_price > self.highest_price:
            self.highest_price = current_price

        # 计算当前盈利
        profit_percent = (current_price - entry_price) / entry_price

        # 确定当前应该使用的级别
        new_level = -1
        for i, (profit_threshold, _) in enumerate(self.stop_levels):
            if profit_percent >= profit_threshold:
                new_level = i

        # 如果级别提升，记录日志
        if new_level > self.current_level:
            self.current_level = new_level
            profit_threshold, trailing_distance = self.stop_levels[new_level]
            logger.info(
                f"移动止损级别提升至 Level {new_level + 1} - "
                f"盈利: {profit_percent*100:.2f}%, "
                f"回撤阈值: {trailing_distance*100:.2f}%"
            )

        # 根据当前级别更新止损价
        if self.current_level >= 0:
            _, trailing_distance = self.stop_levels[self.current_level]
            new_stop = self.highest_price * (1 - trailing_distance)

            # 止损价只能上移
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop

        # 检查是否触发止损
        triggered = current_price <= self.trailing_stop_price
        message = ""

        if triggered:
            message = (
                f"触发多级止损 (Level {self.current_level + 1}) - "
                f"最高价: {self.highest_price:.2f}, "
                f"当前价: {current_price:.2f}, "
                f"止损价: {self.trailing_stop_price:.2f}"
            )
            logger.warning(message)

        return self.trailing_stop_price, triggered, message


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("===== 移动止损测试 =====\n")

    # 创建移动止损管理器
    trailing_stop = TrailingStopManager()

    # 模拟交易场景
    entry_price = 50000
    initial_stop = entry_price * 0.98  # 初始2%止损

    trailing_stop.initialize(entry_price, initial_stop)

    # 模拟价格变化
    price_sequence = [
        50000,  # 入场
        50500,  # 上涨1%
        51000,  # 上涨2% - 激活移动止损
        51500,  # 上涨3%
        52000,  # 上涨4%
        51800,  # 小幅回撤
        52500,  # 继续上涨5%
        51500,  # 大幅回撤 - 可能触发止损
        50500,  # 继续下跌
    ]

    print(f"入场价格: ${entry_price:.2f}")
    print(f"初始止损: ${initial_stop:.2f}\n")

    for i, price in enumerate(price_sequence):
        print(f"--- 步骤 {i + 1}: 价格 ${price:.2f} ---")

        stop_price, triggered, message = trailing_stop.update(price, entry_price)
        status = trailing_stop.get_status(price, entry_price)

        print(f"止损价: ${stop_price:.2f}")
        print(f"已激活: {status['is_activated']}")
        print(f"最高价: ${status['highest_price']:.2f}")
        print(f"当前盈利: {status['profit_percent']:.2f}%")
        print(f"距止损: {status['distance_to_stop_percent']:.2f}%")

        if triggered:
            print(f"⚠️  {message}")
            break

        print()

    print("\n===== 多级移动止损测试 =====\n")

    # 创建多级移动止损管理器
    multi_trailing = MultiLevelTrailingStop()
    multi_trailing.initialize(entry_price, initial_stop)

    # 模拟大幅上涨然后回撤的场景
    price_sequence_2 = [
        50000,  # 入场
        51000,  # +2%
        52500,  # +5% - 进入Level 2
        55000,  # +10% - 进入Level 3
        60000,  # +20% - 进入Level 4
        59000,  # 小幅回撤
        58500,  # 继续回撤
    ]

    print(f"入场价格: ${entry_price:.2f}\n")

    for i, price in enumerate(price_sequence_2):
        profit = (price - entry_price) / entry_price * 100
        print(f"步骤 {i + 1}: 价格 ${price:.2f} (盈利 {profit:.2f}%)")

        stop_price, triggered, message = multi_trailing.update(price, entry_price)
        print(f"  止损价: ${stop_price:.2f}")

        if triggered:
            print(f"  ⚠️  {message}")
            break
