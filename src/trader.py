"""
交易执行模拟器 - 模拟真实交易环境（含手续费、滑点等）
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class Position:
    """持仓信息"""

    def __init__(self):
        self.is_open = False
        self.entry_price = 0.0
        self.entry_time = None
        self.quantity = 0.0
        self.side = None
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def open(self, price: float, quantity: float, timestamp: datetime, stop_loss: float = 0, take_profit: float = 0):
        """开仓"""
        self.is_open = True
        self.entry_price = price
        self.entry_time = timestamp
        self.quantity = quantity
        self.side = OrderSide.BUY
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def close(self):
        """平仓"""
        self.is_open = False
        self.entry_price = 0.0
        self.entry_time = None
        self.quantity = 0.0
        self.side = None
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def get_unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        if not self.is_open:
            return 0.0
        return (current_price - self.entry_price) * self.quantity

    def get_unrealized_pnl_percent(self, current_price: float) -> float:
        """计算未实现盈亏百分比"""
        if not self.is_open or self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * 100


class TradingSimulator:
    """交易模拟器"""

    def __init__(self, config: dict):
        """
        初始化交易模拟器

        Args:
            config: 配置参数
        """
        self.config = config
        self.initial_capital = config['initial_capital']
        self.cash = self.initial_capital
        self.position = Position()

        # 交易参数
        self.fee_rate = config.get('fee_rate', 0.001)  # 0.1%
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 0.05%
        self.min_trade_amount = config.get('min_trade_amount', 10)  # 最小交易金额 USDT
        self.max_position_percent = config.get('max_position_percent', 0.95)  # 最大仓位比例

        # 止损止盈
        self.stop_loss_percent = config.get('stop_loss_percent', 0.02)  # 2%止损
        self.take_profit_percent = config.get('take_profit_percent', 0.05)  # 5%止盈
        self.use_trailing_stop = config.get('use_trailing_stop', False)  # 是否使用移动止损

        # 持仓管理
        self.max_holding_minutes = config.get('max_holding_minutes', 240)  # 最大持仓时间（分钟）

        # 交易记录
        self.trade_history: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # 统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_capital = self.initial_capital
        self.max_drawdown = 0.0

    def get_total_equity(self, current_price: float = 0) -> float:
        """获取总权益"""
        if self.position.is_open and current_price > 0:
            return self.cash + self.position.quantity * current_price
        return self.cash

    def can_buy(self, price: float) -> Tuple[bool, str]:
        """
        检查是否可以买入

        Returns:
            (是否可以买入, 原因)
        """
        if self.position.is_open:
            return False, "已有持仓"

        max_buy_amount = self.cash * self.max_position_percent
        if max_buy_amount < self.min_trade_amount:
            return False, f"可用资金不足（需要至少{self.min_trade_amount} USDT）"

        return True, "OK"

    def can_sell(self) -> Tuple[bool, str]:
        """
        检查是否可以卖出

        Returns:
            (是否可以卖出, 原因)
        """
        if not self.position.is_open:
            return False, "无持仓"

        return True, "OK"

    def execute_buy(
        self,
        price: float,
        timestamp: datetime,
        signal_strength: str = "中"
    ) -> Optional[Dict]:
        """
        执行买入

        Args:
            price: 买入价格
            timestamp: 时间戳
            signal_strength: 信号强度

        Returns:
            交易记录或None
        """
        can_buy, reason = self.can_buy(price)
        if not can_buy:
            logger.warning(f"无法买入: {reason}")
            return None

        # 根据信号强度决定仓位比例
        if signal_strength == "强":
            position_percent = 0.95
        elif signal_strength == "中":
            position_percent = 0.70
        else:  # 弱
            position_percent = 0.50

        # 计算买入金额
        buy_amount = self.cash * position_percent

        # 应用滑点（买入时价格上涨）
        actual_price = price * (1 + self.slippage_rate)

        # 计算数量
        quantity = buy_amount / actual_price

        # 计算手续费
        fee = buy_amount * self.fee_rate

        # 总成本
        total_cost = buy_amount + fee

        if total_cost > self.cash:
            logger.warning(f"资金不足，无法买入")
            return None

        # 计算止损止盈价格
        stop_loss_price = actual_price * (1 - self.stop_loss_percent)
        take_profit_price = actual_price * (1 + self.take_profit_percent)

        # 开仓
        self.position.open(actual_price, quantity, timestamp, stop_loss_price, take_profit_price)

        # 更新现金
        self.cash -= total_cost

        # 记录交易
        trade_record = {
            'timestamp': timestamp,
            'side': 'BUY',
            'price': actual_price,
            'quantity': quantity,
            'amount': buy_amount,
            'fee': fee,
            'total_cost': total_cost,
            'cash_after': self.cash,
            'signal_strength': signal_strength
        }

        self.trade_history.append(trade_record)
        self.total_trades += 1

        logger.info(f"买入成功: 价格={actual_price:.2f}, 数量={quantity:.6f}, 手续费={fee:.2f}")

        return trade_record

    def execute_sell(
        self,
        price: float,
        timestamp: datetime,
        reason: str = "信号卖出"
    ) -> Optional[Dict]:
        """
        执行卖出

        Args:
            price: 卖出价格
            timestamp: 时间戳
            reason: 卖出原因

        Returns:
            交易记录或None
        """
        can_sell, msg = self.can_sell()
        if not can_sell:
            logger.warning(f"无法卖出: {msg}")
            return None

        # 应用滑点（卖出时价格下跌）
        actual_price = price * (1 - self.slippage_rate)

        # 计算卖出金额
        sell_amount = self.position.quantity * actual_price

        # 计算手续费
        fee = sell_amount * self.fee_rate

        # 实际收入
        net_proceeds = sell_amount - fee

        # 计算盈亏
        cost_basis = self.position.entry_price * self.position.quantity
        pnl = sell_amount - cost_basis - fee
        pnl_percent = pnl / cost_basis * 100

        # 更新现金
        self.cash += net_proceeds

        # 更新统计
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # 记录交易
        trade_record = {
            'timestamp': timestamp,
            'side': 'SELL',
            'price': actual_price,
            'quantity': self.position.quantity,
            'amount': sell_amount,
            'fee': fee,
            'net_proceeds': net_proceeds,
            'cash_after': self.cash,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'holding_time': (timestamp - self.position.entry_time).total_seconds() / 60,  # 分钟
            'reason': reason
        }

        self.trade_history.append(trade_record)

        logger.info(f"卖出成功: 价格={actual_price:.2f}, 盈亏={pnl:.2f} ({pnl_percent:.2f}%), 原因={reason}")

        # 平仓
        self.position.close()

        return trade_record

    def check_stop_loss_take_profit(
        self,
        current_price: float,
        timestamp: datetime
    ) -> Optional[Dict]:
        """
        检查止损止盈

        Returns:
            如果触发止损/止盈，返回交易记录，否则返回None
        """
        if not self.position.is_open:
            return None

        # 检查止损
        if current_price <= self.position.stop_loss:
            logger.warning(f"触发止损: 当前价格={current_price}, 止损价={self.position.stop_loss}")
            return self.execute_sell(current_price, timestamp, "止损")

        # 检查止盈
        if current_price >= self.position.take_profit:
            logger.info(f"触发止盈: 当前价格={current_price}, 止盈价={self.position.take_profit}")
            return self.execute_sell(current_price, timestamp, "止盈")

        # 检查最大持仓时间
        holding_minutes = (timestamp - self.position.entry_time).total_seconds() / 60
        if holding_minutes >= self.max_holding_minutes:
            logger.info(f"达到最大持仓时间: {holding_minutes:.0f}分钟")
            return self.execute_sell(current_price, timestamp, "超时平仓")

        return None

    def update_equity_curve(self, timestamp: datetime, current_price: float):
        """更新权益曲线"""
        total_equity = self.get_total_equity(current_price)

        # 更新最大权益
        if total_equity > self.max_capital:
            self.max_capital = total_equity

        # 计算回撤
        if self.max_capital > 0:
            drawdown = (self.max_capital - total_equity) / self.max_capital * 100
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        else:
            drawdown = 0

        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': self.position.quantity * current_price if self.position.is_open else 0,
            'total_equity': total_equity,
            'drawdown': drawdown
        })

    def get_performance_stats(self) -> Dict:
        """获取绩效统计"""
        final_capital = self.cash
        total_return = (
            (final_capital - self.initial_capital) / self.initial_capital * 100
            if self.initial_capital > 0 else 0
        )

        stats = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': self.max_drawdown,
            'avg_holding_time': 0
        }

        if self.total_trades == 0:
            return stats

        # 胜率
        win_rate = self.winning_trades / self.total_trades * 100

        # 盈亏比
        winning_pnl = sum([t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_pnl = abs(sum([t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0]))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0

        # 平均持仓时间
        holding_times = [t['holding_time'] for t in self.trade_history if 'holding_time' in t]
        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

        stats.update({
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_holding_time': avg_holding_time
        })

        return stats


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = {
        'initial_capital': 1000,
        'fee_rate': 0.001,
        'slippage_rate': 0.0005,
        'stop_loss_percent': 0.02,
        'take_profit_percent': 0.05
    }

    simulator = TradingSimulator(config)

    # 模拟交易
    timestamp = datetime.now()
    simulator.execute_buy(50000, timestamp, "强")
    print(f"买入后现金: {simulator.cash:.2f}")

    simulator.execute_sell(51000, timestamp, "止盈")
    print(f"卖出后现金: {simulator.cash:.2f}")

    stats = simulator.get_performance_stats()
    print("\n绩效统计:")
    for key, value in stats.items():
        print(f"{key}: {value}")
