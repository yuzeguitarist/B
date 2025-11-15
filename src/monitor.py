"""
输出与监控模块 - 交易日志、统计报告、可视化
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestMonitor:
    """回测监控器"""

    def __init__(self, output_dir: str = "./results"):
        """
        初始化监控器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志记录
        self.trade_logs: List[Dict] = []
        self.hourly_stats: List[Dict] = []

        # 创建日志文件
        self.log_file = self.output_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._setup_file_logger()

    def _setup_file_logger(self):
        """设置文件日志"""
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def log_trade(
        self,
        timestamp: datetime,
        action: str,
        price: float,
        quantity: float,
        amount: float,
        reason: str,
        pnl: float = 0,
        pnl_percent: float = 0,
        capital_after: float = 0
    ):
        """
        记录交易日志

        Args:
            timestamp: 时间戳
            action: 操作（BUY/SELL）
            price: 价格
            quantity: 数量
            amount: 金额
            reason: 原因
            pnl: 盈亏
            pnl_percent: 盈亏百分比
            capital_after: 交易后资金
        """
        log_entry = {
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'quantity': quantity,
            'amount': amount,
            'reason': reason,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'capital_after': capital_after
        }

        self.trade_logs.append(log_entry)

        # 写入日志
        if action == 'BUY':
            logger.info(
                f"【买入】价格={price:.2f}, 数量={quantity:.6f}, "
                f"金额={amount:.2f}, 原因={reason}, 资金余额={capital_after:.2f}"
            )
        elif action == 'SELL':
            logger.info(
                f"【卖出】价格={price:.2f}, 数量={quantity:.6f}, "
                f"金额={amount:.2f}, 盈亏={pnl:.2f} ({pnl_percent:.2f}%), "
                f"原因={reason}, 资金余额={capital_after:.2f}"
            )

    def log_hourly_stats(
        self,
        hour: int,
        timestamp: datetime,
        trades_count: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        capital: float,
        max_drawdown: float
    ):
        """
        记录每小时统计

        Args:
            hour: 第几小时
            timestamp: 时间戳
            trades_count: 交易次数
            winning_trades: 盈利次数
            losing_trades: 亏损次数
            total_pnl: 总盈亏
            capital: 当前资金
            max_drawdown: 最大回撤
        """
        stats = {
            'hour': hour,
            'timestamp': timestamp,
            'trades_count': trades_count,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'capital': capital,
            'max_drawdown': max_drawdown
        }

        self.hourly_stats.append(stats)

        logger.info(
            f"\n{'='*60}\n"
            f"第{hour}小时统计 ({timestamp.strftime('%Y-%m-%d %H:%M')})\n"
            f"{'='*60}\n"
            f"交易次数: {trades_count} (盈利: {winning_trades}, 亏损: {losing_trades})\n"
            f"累计盈亏: {total_pnl:.2f} USDT\n"
            f"当前资金: {capital:.2f} USDT\n"
            f"最大回撤: {max_drawdown:.2f}%\n"
            f"{'='*60}"
        )

    def generate_summary_report(
        self,
        initial_capital: float,
        final_capital: float,
        trade_history: List[Dict],
        equity_curve: List[Dict],
        duration_hours: float
    ) -> Dict:
        """
        生成总结报告

        Args:
            initial_capital: 初始资金
            final_capital: 最终资金
            trade_history: 交易历史
            equity_curve: 权益曲线
            duration_hours: 持续时间（小时）

        Returns:
            报告字典
        """
        logger.info("\n" + "="*60)
        logger.info("72小时回测总结报告")
        logger.info("="*60)

        # 基础统计
        total_return = final_capital - initial_capital
        total_return_percent = (total_return / initial_capital) * 100

        # 过滤出卖出交易
        sell_trades = [t for t in trade_history if t.get('side') == 'SELL']

        total_trades = len(sell_trades)
        if total_trades == 0:
            logger.warning("没有完成的交易，无法生成详细报告")
            return self._create_empty_report(initial_capital, final_capital, duration_hours)

        winning_trades = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in sell_trades if t.get('pnl', 0) < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # 盈亏统计
        winning_pnl = sum([t['pnl'] for t in sell_trades if t.get('pnl', 0) > 0])
        losing_pnl = abs(sum([t['pnl'] for t in sell_trades if t.get('pnl', 0) < 0]))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

        # 平均盈亏
        avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
        avg_loss = losing_pnl / losing_trades if losing_trades > 0 else 0
        avg_pnl = total_return / total_trades if total_trades > 0 else 0

        # 最大单笔盈亏
        max_win = max([t.get('pnl', 0) for t in sell_trades]) if sell_trades else 0
        max_loss = min([t.get('pnl', 0) for t in sell_trades]) if sell_trades else 0

        # 最大回撤
        if equity_curve:
            max_drawdown = max([e['drawdown'] for e in equity_curve])
        else:
            max_drawdown = 0

        # 夏普比率（简化版，假设无风险利率为0）
        if sell_trades:
            returns = [t.get('pnl', 0) / initial_capital for t in sell_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # 平均持仓时间
        holding_times = [t.get('holding_time', 0) for t in sell_trades]
        avg_holding_time = np.mean(holding_times) if holding_times else 0

        report = {
            'duration_hours': duration_hours,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': avg_pnl,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_holding_time': avg_holding_time
        }

        # 输出报告
        self._print_summary_report(report)

        # 保存报告
        self._save_report(report, trade_history, equity_curve)

        # 失败交易分析
        self._analyze_failed_trades(sell_trades)

        return report

    def _create_empty_report(self, initial_capital: float, final_capital: float, duration_hours: float) -> Dict:
        """创建空报告（无交易时）"""
        return {
            'duration_hours': duration_hours,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': final_capital - initial_capital,
            'total_return_percent': ((final_capital - initial_capital) / initial_capital) * 100,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_pnl': 0,
            'max_win': 0,
            'max_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_holding_time': 0
        }

    def _print_summary_report(self, report: Dict):
        """打印总结报告"""
        logger.info(f"\n回测时长: {report['duration_hours']:.1f} 小时")
        logger.info(f"\n资金情况:")
        logger.info(f"  初始资金: {report['initial_capital']:.2f} USDT")
        logger.info(f"  最终资金: {report['final_capital']:.2f} USDT")
        logger.info(f"  总收益: {report['total_return']:.2f} USDT ({report['total_return_percent']:.2f}%)")

        logger.info(f"\n交易统计:")
        logger.info(f"  总交易次数: {report['total_trades']}")
        logger.info(f"  盈利次数: {report['winning_trades']}")
        logger.info(f"  亏损次数: {report['losing_trades']}")
        logger.info(f"  胜率: {report['win_rate']:.2f}%")

        logger.info(f"\n盈亏分析:")
        logger.info(f"  盈亏比: {report['profit_factor']:.2f}")
        logger.info(f"  平均盈利: {report['avg_win']:.2f} USDT")
        logger.info(f"  平均亏损: {report['avg_loss']:.2f} USDT")
        logger.info(f"  平均每笔: {report['avg_pnl']:.2f} USDT")
        logger.info(f"  最大单笔盈利: {report['max_win']:.2f} USDT")
        logger.info(f"  最大单笔亏损: {report['max_loss']:.2f} USDT")

        logger.info(f"\n风险指标:")
        logger.info(f"  最大回撤: {report['max_drawdown']:.2f}%")
        logger.info(f"  夏普比率: {report['sharpe_ratio']:.2f}")

        logger.info(f"\n持仓统计:")
        logger.info(f"  平均持仓时间: {report['avg_holding_time']:.1f} 分钟")

        logger.info("="*60)

    def _save_report(self, report: Dict, trade_history: List[Dict], equity_curve: List[Dict]):
        """保存报告到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存JSON报告
        report_file = self.output_dir / f"report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"\n报告已保存: {report_file}")

        # 保存交易历史
        if trade_history:
            trades_df = pd.DataFrame(trade_history)
            trades_file = self.output_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False, encoding='utf-8')
            logger.info(f"交易历史已保存: {trades_file}")

        # 保存权益曲线
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            equity_file = self.output_dir / f"equity_{timestamp}.csv"
            equity_df.to_csv(equity_file, index=False, encoding='utf-8')
            logger.info(f"权益曲线已保存: {equity_file}")

    def _analyze_failed_trades(self, sell_trades: List[Dict]):
        """分析失败交易"""
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]

        if not losing_trades:
            logger.info("\n没有亏损交易")
            return

        logger.info("\n" + "="*60)
        logger.info("失败交易分析")
        logger.info("="*60)

        # 按原因分类
        reasons = {}
        for trade in losing_trades:
            reason = trade.get('reason', '未知')
            if reason not in reasons:
                reasons[reason] = []
            reasons[reason].append(trade)

        logger.info(f"\n亏损交易原因分布:")
        for reason, trades in reasons.items():
            total_loss = sum([t['pnl'] for t in trades])
            logger.info(f"  {reason}: {len(trades)}次, 总亏损: {total_loss:.2f} USDT")

        # 最大亏损交易
        worst_trade = min(losing_trades, key=lambda x: x.get('pnl', 0))
        logger.info(f"\n最大亏损交易:")
        logger.info(f"  时间: {worst_trade.get('timestamp')}")
        logger.info(f"  价格: {worst_trade.get('price', 0):.2f}")
        logger.info(f"  亏损: {worst_trade.get('pnl', 0):.2f} USDT ({worst_trade.get('pnl_percent', 0):.2f}%)")
        logger.info(f"  原因: {worst_trade.get('reason', '未知')}")

        logger.info("="*60)

    def plot_equity_curve(self, equity_curve: List[Dict]):
        """
        绘制权益曲线（需要matplotlib）

        Args:
            equity_curve: 权益曲线数据
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if not equity_curve:
                logger.warning("权益曲线数据为空，无法绘图")
                return

            df = pd.DataFrame(equity_curve)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # 权益曲线
            ax1.plot(df['timestamp'], df['total_equity'], label='Total Equity', linewidth=2)
            ax1.set_ylabel('Capital (USDT)', fontsize=12)
            ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 回撤曲线
            ax2.fill_between(df['timestamp'], 0, -df['drawdown'], color='red', alpha=0.3)
            ax2.plot(df['timestamp'], -df['drawdown'], color='red', linewidth=2)
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # 格式化x轴
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)

            plt.tight_layout()

            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_file = self.output_dir / f"equity_curve_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            logger.info(f"权益曲线图已保存: {chart_file}")

            plt.close()

        except ImportError:
            logger.warning("matplotlib未安装，无法绘制图表")
        except Exception as e:
            logger.error(f"绘图失败: {e}")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    monitor = BacktestMonitor()

    # 模拟交易日志
    monitor.log_trade(
        timestamp=datetime.now(),
        action='BUY',
        price=50000,
        quantity=0.02,
        amount=1000,
        reason='RSI超卖',
        capital_after=0
    )

    monitor.log_trade(
        timestamp=datetime.now(),
        action='SELL',
        price=51000,
        quantity=0.02,
        amount=1020,
        reason='止盈',
        pnl=20,
        pnl_percent=2.0,
        capital_after=1020
    )

    # 生成报告
    report = monitor.generate_summary_report(
        initial_capital=1000,
        final_capital=1020,
        trade_history=[
            {'side': 'BUY', 'amount': 1000},
            {'side': 'SELL', 'pnl': 20, 'pnl_percent': 2.0, 'holding_time': 30, 'reason': '止盈'}
        ],
        equity_curve=[],
        duration_hours=72
    )

    print(f"\n报告: {report}")
