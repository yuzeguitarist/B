"""
分批止盈模块 - 在不同盈利水平分批卖出锁定利润
"""
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class PartialProfitManager:
    """分批止盈管理器"""

    def __init__(self, config: dict = None):
        """
        初始化分批止盈管理器

        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()

        # 定义分批止盈级别
        # 格式：[(盈利目标%, 卖出比例), ...]
        self.profit_levels = self.config.get('profit_levels', [
            (0.30, 0.33),  # 盈利30%时卖出1/3
            (0.60, 0.50),  # 盈利60%时卖出剩余的1/2（即总仓位的1/3）
            # 剩余1/3使用移动止损
        ])

        # 状态跟踪
        self.initial_quantity = 0  # 初始持仓数量
        self.remaining_quantity = 0  # 剩余持仓数量
        self.executed_levels = []  # 已执行的级别
        self.total_realized_profit = 0  # 已实现利润

    @staticmethod
    def _get_default_config() -> dict:
        """默认配置"""
        return {
            'use_partial_profit': True,  # 是否启用分批止盈
            'profit_levels': [
                (0.30, 0.33),  # 盈利30%，卖出33%
                (0.60, 0.50),  # 盈利60%，卖出50%（基于剩余仓位）
            ],
        }

    def initialize(self, quantity: float):
        """
        初始化分批止盈（开仓时调用）

        Args:
            quantity: 初始持仓数量
        """
        self.initial_quantity = quantity
        self.remaining_quantity = quantity
        self.executed_levels = []
        self.total_realized_profit = 0

        logger.info(f"分批止盈初始化 - 初始持仓: {quantity:.6f}")

    def check_and_execute(
        self,
        current_price: float,
        entry_price: float
    ) -> Tuple[Optional[int], Optional[float], str]:
        """
        检查是否触发分批止盈，并计算应该卖出的数量

        Args:
            current_price: 当前价格
            entry_price: 入场价格

        Returns:
            (触发的级别索引, 卖出数量, 说明信息)
            如果没有触发，返回 (None, None, "")
        """
        if self.remaining_quantity <= 0:
            return None, None, "已无剩余持仓"

        # 计算当前盈利百分比
        profit_percent = (current_price - entry_price) / entry_price

        # 检查每个级别
        for level_idx, (target_profit, sell_ratio) in enumerate(self.profit_levels):
            # 如果该级别已执行，跳过
            if level_idx in self.executed_levels:
                continue

            # 如果达到盈利目标
            if profit_percent >= target_profit:
                # 计算卖出数量（基于剩余持仓）
                sell_quantity = self.remaining_quantity * sell_ratio

                # 标记该级别已执行
                self.executed_levels.append(level_idx)

                # 更新剩余持仓
                self.remaining_quantity -= sell_quantity

                # 计算该笔已实现利润
                realized_profit = sell_quantity * (current_price - entry_price)
                self.total_realized_profit += realized_profit

                message = (
                    f"触发分批止盈 Level {level_idx + 1} - "
                    f"盈利: {profit_percent*100:.2f}%, "
                    f"卖出: {sell_quantity:.6f} ({sell_ratio*100:.0f}%), "
                    f"剩余: {self.remaining_quantity:.6f} ({self.get_remaining_percent()*100:.0f}%), "
                    f"本次盈利: ${realized_profit:.2f}"
                )

                logger.info(message)

                return level_idx, sell_quantity, message

        return None, None, ""

    def get_remaining_percent(self) -> float:
        """
        获取剩余持仓占初始持仓的百分比

        Returns:
            剩余持仓比例（0-1）
        """
        if self.initial_quantity <= 0:
            return 0
        return self.remaining_quantity / self.initial_quantity

    def get_status(self, current_price: float, entry_price: float) -> Dict:
        """
        获取分批止盈状态信息

        Args:
            current_price: 当前价格
            entry_price: 入场价格

        Returns:
            状态信息字典
        """
        profit_percent = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        # 计算下一个未执行的级别
        next_level = None
        for level_idx, (target_profit, sell_ratio) in enumerate(self.profit_levels):
            if level_idx not in self.executed_levels:
                next_level = {
                    'level': level_idx + 1,
                    'target_profit_percent': target_profit * 100,
                    'sell_ratio': sell_ratio * 100,
                    'distance_percent': (target_profit - profit_percent) * 100
                }
                break

        return {
            'initial_quantity': self.initial_quantity,
            'remaining_quantity': self.remaining_quantity,
            'remaining_percent': self.get_remaining_percent() * 100,
            'executed_levels': len(self.executed_levels),
            'total_levels': len(self.profit_levels),
            'current_profit_percent': profit_percent * 100,
            'total_realized_profit': self.total_realized_profit,
            'next_level': next_level,
        }

    def is_fully_executed(self) -> bool:
        """
        检查是否所有级别都已执行

        Returns:
            是否全部执行完毕
        """
        return len(self.executed_levels) >= len(self.profit_levels)

    def reset(self):
        """重置分批止盈状态"""
        self.initial_quantity = 0
        self.remaining_quantity = 0
        self.executed_levels = []
        self.total_realized_profit = 0
        logger.info("分批止盈已重置")


class DynamicPartialProfitManager:
    """动态分批止盈管理器（根据市场情况调整止盈目标）"""

    def __init__(self, config: dict = None):
        """
        初始化动态分批止盈管理器

        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()

        # 基础止盈级别
        self.base_levels = [
            (0.20, 0.25),  # 盈利20%，卖出25%
            (0.40, 0.33),  # 盈利40%，卖出33%
            (0.80, 0.50),  # 盈利80%，卖出50%
        ]

        # 状态
        self.initial_quantity = 0
        self.remaining_quantity = 0
        self.executed_levels = []

    @staticmethod
    def _get_default_config() -> dict:
        """默认配置"""
        return {
            'use_dynamic_partial_profit': False,  # 是否使用动态分批止盈
            'volatility_adjustment': True,        # 是否根据波动率调整
        }

    def adjust_levels_by_volatility(self, volatility: float) -> List[Tuple[float, float]]:
        """
        根据波动率调整止盈级别

        Args:
            volatility: 当前波动率

        Returns:
            调整后的止盈级别
        """
        # 如果波动率高，降低止盈目标（快速获利了结）
        # 如果波动率低，提高止盈目标（持有更长时间）

        if volatility > 0.05:  # 高波动
            multiplier = 0.7  # 降低目标30%
        elif volatility > 0.03:  # 中等波动
            multiplier = 0.85  # 降低目标15%
        elif volatility < 0.015:  # 低波动
            multiplier = 1.3  # 提高目标30%
        else:  # 正常波动
            multiplier = 1.0

        adjusted_levels = [
            (profit_target * multiplier, sell_ratio)
            for profit_target, sell_ratio in self.base_levels
        ]

        logger.info(f"根据波动率 {volatility*100:.2f}% 调整止盈级别 (倍数: {multiplier})")

        return adjusted_levels

    def initialize(self, quantity: float, volatility: Optional[float] = None):
        """
        初始化动态分批止盈

        Args:
            quantity: 初始持仓数量
            volatility: 当前波动率（可选）
        """
        self.initial_quantity = quantity
        self.remaining_quantity = quantity
        self.executed_levels = []

        # 根据波动率调整级别
        if volatility is not None and self.config.get('volatility_adjustment'):
            self.profit_levels = self.adjust_levels_by_volatility(volatility)
        else:
            self.profit_levels = self.base_levels

        logger.info(f"动态分批止盈初始化 - 持仓: {quantity:.6f}, 级别: {self.profit_levels}")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("===== 分批止盈测试 =====\n")

    # 创建分批止盈管理器
    partial_profit = PartialProfitManager()

    # 模拟交易场景
    entry_price = 50000
    initial_quantity = 1.0  # 1个BTC

    partial_profit.initialize(initial_quantity)

    # 模拟价格上涨过程
    price_sequence = [
        (50000, "入场"),
        (55000, "上涨10%"),
        (60000, "上涨20%"),
        (65000, "上涨30% - 触发第一次止盈"),
        (70000, "上涨40%"),
        (75000, "上涨50%"),
        (80000, "上涨60% - 触发第二次止盈"),
        (85000, "上涨70%"),
        (90000, "上涨80%"),
    ]

    print(f"入场价格: ${entry_price:.2f}")
    print(f"初始持仓: {initial_quantity} BTC\n")

    total_sold = 0
    total_proceeds = 0

    for price, description in price_sequence:
        profit_pct = (price - entry_price) / entry_price * 100
        print(f"--- {description} (${price:.2f}, 盈利 {profit_pct:.1f}%) ---")

        # 检查是否触发止盈
        level, sell_qty, message = partial_profit.check_and_execute(price, entry_price)

        if level is not None:
            print(f"✓ {message}")
            total_sold += sell_qty
            total_proceeds += sell_qty * price

        # 显示状态
        status = partial_profit.get_status(price, entry_price)
        print(f"剩余持仓: {status['remaining_quantity']:.6f} BTC ({status['remaining_percent']:.1f}%)")

        if status['next_level']:
            next_lvl = status['next_level']
            print(f"下一级别: Level {next_lvl['level']} "
                  f"(目标 +{next_lvl['target_profit_percent']:.0f}%, "
                  f"还需 +{next_lvl['distance_percent']:.1f}%)")

        print()

    # 最终统计
    print("===== 最终统计 =====")
    print(f"总卖出: {total_sold:.6f} BTC")
    print(f"卖出收益: ${total_proceeds:.2f}")
    print(f"剩余持仓: {partial_profit.remaining_quantity:.6f} BTC")
    print(f"已实现利润: ${partial_profit.total_realized_profit:.2f}")

    # 测试动态分批止盈
    print("\n\n===== 动态分批止盈测试 =====\n")

    dynamic_partial = DynamicPartialProfitManager()

    # 测试不同波动率下的级别调整
    for vol in [0.01, 0.02, 0.04, 0.06]:
        print(f"波动率 {vol*100:.0f}%:")
        levels = dynamic_partial.adjust_levels_by_volatility(vol)
        for i, (target, ratio) in enumerate(levels):
            print(f"  Level {i+1}: 盈利 {target*100:.1f}% 时卖出 {ratio*100:.0f}%")
        print()
