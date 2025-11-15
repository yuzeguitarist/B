"""
风险控制模块 - 实施各种风险管理规则
"""
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """风险管理器"""

    def __init__(self, config: dict):
        """
        初始化风险管理器

        Args:
            config: 风险控制配置
        """
        self.config = config

        # 单日最大亏损限制
        self.max_daily_loss_percent = config.get('max_daily_loss_percent', 0.05)  # 5%

        # 连续亏损熔断
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)

        # 急跌保护
        self.flash_crash_threshold = config.get('flash_crash_threshold', 0.05)  # 5%
        self.flash_crash_period = config.get('flash_crash_period', 5)  # 5分钟

        # 最大回撤限制
        self.max_drawdown_percent = config.get('max_drawdown_percent', 0.10)  # 10%

        # 状态跟踪
        self.consecutive_losses = 0
        self.daily_start_capital = 0
        self.current_date = None
        self.is_trading_halted = False
        self.halt_reason = ""
        self.halt_until = None

    def check_risk_before_trade(
        self,
        current_capital: float,
        initial_capital: float,
        recent_trades: list,
        current_price: float,
        price_history: pd.DataFrame,
        timestamp: datetime
    ) -> Tuple[bool, str]:
        """
        交易前风险检查

        Args:
            current_capital: 当前资金
            initial_capital: 初始资金
            recent_trades: 最近交易记录
            current_price: 当前价格
            price_history: 价格历史数据（最近一段时间）
            timestamp: 当前时间

        Returns:
            (是否允许交易, 原因)
        """
        # 检查是否在熔断期
        if self.is_trading_halted:
            if self.halt_until and timestamp < self.halt_until:
                return False, f"交易已暂停: {self.halt_reason}"
            else:
                # 解除熔断
                self.is_trading_halted = False
                self.halt_reason = ""
                logger.info("交易暂停已解除")

        # 1. 检查单日亏损限制
        passed, reason = self._check_daily_loss(current_capital, timestamp)
        if not passed:
            self._halt_trading(reason, timestamp, hours=24)
            return False, reason

        # 2. 检查连续亏损熔断
        passed, reason = self._check_consecutive_losses(recent_trades)
        if not passed:
            self._halt_trading(reason, timestamp, hours=2)
            return False, reason

        # 3. 检查急跌保护
        passed, reason = self._check_flash_crash(price_history)
        if not passed:
            self._halt_trading(reason, timestamp, hours=1)
            return False, reason

        # 4. 检查最大回撤
        passed, reason = self._check_max_drawdown(current_capital, initial_capital)
        if not passed:
            self._halt_trading(reason, timestamp, hours=24)
            return False, reason

        return True, "风险检查通过"

    def _check_daily_loss(self, current_capital: float, timestamp: datetime) -> Tuple[bool, str]:
        """检查单日亏损限制"""
        current_date = timestamp.date()

        # 如果是新的一天，重置起始资金
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_start_capital = current_capital
            logger.info(f"新的交易日开始，起始资金: {self.daily_start_capital:.2f}")

        # 计算今日亏损
        daily_loss = self.daily_start_capital - current_capital
        daily_loss_percent = daily_loss / self.daily_start_capital if self.daily_start_capital > 0 else 0

        if daily_loss_percent > self.max_daily_loss_percent:
            reason = f"触发单日最大亏损限制: 亏损{daily_loss_percent*100:.2f}% (限制{self.max_daily_loss_percent*100}%)"
            logger.error(reason)
            return False, reason

        return True, "单日亏损检查通过"

    def _check_consecutive_losses(self, recent_trades: list) -> Tuple[bool, str]:
        """检查连续亏损熔断"""
        if len(recent_trades) < 2:
            return True, "交易次数不足"

        # 统计最近的连续亏损次数
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            if 'pnl' in trade:
                if trade['pnl'] < 0:
                    consecutive_losses += 1
                else:
                    break

        self.consecutive_losses = consecutive_losses

        if consecutive_losses >= self.max_consecutive_losses:
            reason = f"触发连续亏损熔断: 连续亏损{consecutive_losses}次 (限制{self.max_consecutive_losses}次)"
            logger.error(reason)
            return False, reason

        return True, "连续亏损检查通过"

    def _check_flash_crash(self, price_history: pd.DataFrame) -> Tuple[bool, str]:
        """检查急跌保护"""
        if len(price_history) < self.flash_crash_period:
            return True, "价格历史不足"

        # 获取最近N分钟的价格数据
        recent_prices = price_history.tail(self.flash_crash_period)

        # 计算价格变化
        price_change = (recent_prices['close'].iloc[-1] - recent_prices['close'].iloc[0]) / recent_prices['close'].iloc[0]

        # 检查是否急跌
        if price_change < -self.flash_crash_threshold:
            reason = f"触发急跌保护: {self.flash_crash_period}分钟内下跌{abs(price_change)*100:.2f}%"
            logger.error(reason)
            return False, reason

        return True, "急跌检查通过"

    def _check_max_drawdown(self, current_capital: float, initial_capital: float) -> Tuple[bool, str]:
        """检查最大回撤限制"""
        # 计算总回撤
        total_drawdown = (initial_capital - current_capital) / initial_capital

        if total_drawdown > self.max_drawdown_percent:
            reason = f"触发最大回撤限制: 回撤{total_drawdown*100:.2f}% (限制{self.max_drawdown_percent*100}%)"
            logger.error(reason)
            return False, reason

        return True, "最大回撤检查通过"

    def _halt_trading(self, reason: str, timestamp: datetime, hours: int = 1):
        """暂停交易"""
        self.is_trading_halted = True
        self.halt_reason = reason
        self.halt_until = timestamp + timedelta(hours=hours)
        logger.warning(f"交易已暂停{hours}小时，原因: {reason}")

    def update_after_trade(self, trade_result: dict):
        """交易后更新风险状态"""
        # 如果是卖出交易，更新连续亏损计数
        if trade_result.get('side') == 'SELL':
            pnl = trade_result.get('pnl', 0)
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0  # 重置连续亏损计数

    def get_position_size_recommendation(
        self,
        current_capital: float,
        risk_level: str,
        signal_strength: str
    ) -> float:
        """
        根据风险等级和信号强度推荐仓位大小

        Args:
            current_capital: 当前资金
            risk_level: 风险等级（低/中/高）
            signal_strength: 信号强度（强/中/弱）

        Returns:
            推荐仓位比例 (0-1)
        """
        # 基础仓位
        base_position = {
            '强': 0.8,
            '中': 0.6,
            '弱': 0.4
        }.get(signal_strength, 0.5)

        # 根据风险等级调整
        risk_multiplier = {
            '低': 1.0,
            '中': 0.7,
            '高': 0.5
        }.get(risk_level, 0.7)

        recommended_position = base_position * risk_multiplier

        # 根据连续亏损情况降低仓位
        if self.consecutive_losses > 0:
            loss_penalty = 0.9 ** self.consecutive_losses  # 每次亏损降低10%
            recommended_position *= loss_penalty

        return min(recommended_position, 0.95)  # 最大95%

    def get_risk_status(self) -> Dict:
        """获取当前风险状态"""
        return {
            'is_trading_halted': self.is_trading_halted,
            'halt_reason': self.halt_reason,
            'halt_until': self.halt_until,
            'consecutive_losses': self.consecutive_losses,
            'daily_start_capital': self.daily_start_capital,
            'current_date': self.current_date
        }


class PositionSizer:
    """仓位管理器"""

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float
    ) -> float:
        """
        凯利公式计算最优仓位

        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            capital: 当前资金

        Returns:
            推荐仓位金额
        """
        if avg_loss == 0 or win_rate == 0:
            return capital * 0.1  # 默认10%

        # 凯利公式: f = (bp - q) / b
        # b = avg_win / avg_loss (盈亏比)
        # p = win_rate (胜率)
        # q = 1 - p (败率)

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_percent = (b * p - q) / b

        # 使用半凯利（更保守）
        kelly_percent = kelly_percent * 0.5

        # 限制在合理范围内
        kelly_percent = max(0.05, min(kelly_percent, 0.5))

        return capital * kelly_percent

    @staticmethod
    def fixed_fractional(capital: float, risk_percent: float = 0.02) -> float:
        """
        固定比例仓位管理

        Args:
            capital: 当前资金
            risk_percent: 风险比例（默认2%）

        Returns:
            推荐仓位金额
        """
        return capital * risk_percent


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = {
        'max_daily_loss_percent': 0.05,
        'max_consecutive_losses': 3,
        'flash_crash_threshold': 0.05,
        'flash_crash_period': 5,
        'max_drawdown_percent': 0.10
    }

    risk_manager = RiskManager(config)

    # 模拟检查
    current_capital = 950  # 从1000跌到950
    initial_capital = 1000
    recent_trades = [
        {'pnl': -10},
        {'pnl': -20},
        {'pnl': -20}  # 连续3次亏损
    ]

    # 创建模拟价格历史
    dates = pd.date_range('2024-01-01', periods=10, freq='1min')
    price_history = pd.DataFrame({
        'timestamp': dates,
        'close': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]  # 持续下跌
    })

    passed, reason = risk_manager.check_risk_before_trade(
        current_capital=current_capital,
        initial_capital=initial_capital,
        recent_trades=recent_trades,
        current_price=91,
        price_history=price_history,
        timestamp=datetime.now()
    )

    print(f"风险检查结果: {passed}")
    print(f"原因: {reason}")
    print(f"\n风险状态: {risk_manager.get_risk_status()}")
