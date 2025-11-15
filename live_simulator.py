"""
实时交易模拟器 - 从当前时刻开始，持续72小时实时拉取数据并交易
"""
import sys
import time
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import signal

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_fetcher import BinanceDataFetcher
from data_cleaner import DataCleaner
from indicators import TechnicalIndicators
from strategy import TradingStrategy, SignalType
from trader import TradingSimulator
from risk_manager import RiskManager
from monitor import BacktestMonitor


class LiveTradingSimulator:
    """实时交易模拟器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化实时模拟器"""
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设置日志
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # 初始化各模块
        self.data_fetcher = BinanceDataFetcher(
            api_key=self.config.get('api_key', ''),
            api_secret=self.config.get('api_secret', '')
        )
        self.data_cleaner = DataCleaner(timeframe=self.config['timeframe'])
        self.strategy = TradingStrategy(self.config.get('strategy', {}))
        self.trader = TradingSimulator(self.config.get('trading', {}))
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        self.monitor = BacktestMonitor(self.config.get('output_dir', './results'))

        # 运行参数
        self.symbol = self.config['symbol']
        self.timeframe = self.config['timeframe']
        self.total_hours = self.config.get('hours', 72)

        # 解析时间间隔（秒）
        self.interval_seconds = self._parse_timeframe(self.config['timeframe'])

        # 历史数据缓存（用于计算技术指标）
        self.history_data = pd.DataFrame()
        self.min_history_bars = 100  # 最少需要100根K线才能开始交易

        # 运行状态
        self.is_running = False
        self.start_time = None
        self.end_time = None

        # 统计
        self.total_iterations = 0
        self.last_hourly_report = None

    def _setup_logging(self):
        """设置日志"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    @staticmethod
    def _parse_timeframe(timeframe: str) -> int:
        """解析时间周期为秒数"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            raise ValueError(f"不支持的时间周期: {timeframe}")

    def _fetch_initial_history(self):
        """获取初始历史数据（用于计算技术指标）"""
        self.logger.info("获取初始历史数据用于技术指标计算...")

        # 获取过去的数据（至少100根K线）
        hours_needed = max(2, self.min_history_bars * self.interval_seconds / 3600)

        df = self.data_fetcher.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self.timeframe,
            hours=int(hours_needed)
        )

        # 清洗数据
        df = self.data_cleaner.clean_data(df)

        # 计算技术指标
        indicator_calculator = TechnicalIndicators(df)
        df = indicator_calculator.calculate_all_indicators(self.config.get('indicators', {}))

        self.history_data = df
        self.logger.info(f"初始历史数据加载完成: {len(df)} 条记录")

    def _fetch_latest_bar(self) -> pd.DataFrame:
        """获取最新的一根K线"""
        try:
            # 获取最近的K线（获取2根以防数据延迟）
            df = self.data_fetcher.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                hours=0.1  # 获取最近几分钟的数据
            )

            if len(df) > 0:
                return df.iloc[-1:]  # 返回最新的一根
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取最新K线失败: {e}")
            return pd.DataFrame()

    def _update_history_data(self, new_bar: pd.DataFrame):
        """更新历史数据缓存"""
        if new_bar.empty:
            return

        # 检查是否是新的K线（避免重复）
        if len(self.history_data) > 0:
            last_timestamp = self.history_data.iloc[-1]['timestamp']
            new_timestamp = new_bar.iloc[0]['timestamp']

            if new_timestamp <= last_timestamp:
                # 不是新K线，可能是当前K线的更新，更新最后一根
                self.history_data.iloc[-1] = new_bar.iloc[0]
                return

        # 添加新K线
        self.history_data = pd.concat([self.history_data, new_bar], ignore_index=True)

        # 只保留最近的1000根K线（节省内存）
        if len(self.history_data) > 1000:
            self.history_data = self.history_data.iloc[-1000:]

        # 重新计算技术指标
        indicator_calculator = TechnicalIndicators(self.history_data)
        self.history_data = indicator_calculator.calculate_all_indicators(
            self.config.get('indicators', {})
        )

    def _process_trading_logic(self):
        """处理交易逻辑"""
        if len(self.history_data) < self.min_history_bars:
            self.logger.debug(f"历史数据不足，当前{len(self.history_data)}根，需要{self.min_history_bars}根")
            return

        current_index = len(self.history_data) - 1
        current_bar = self.history_data.iloc[current_index]
        timestamp = current_bar['timestamp']
        current_price = current_bar['close']

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"时间: {timestamp}, 价格: {current_price:.2f} USDT")

        # 更新权益曲线
        self.trader.update_equity_curve(timestamp, current_price)

        # 检查止损止盈
        if self.trader.position.is_open:
            trade_result = self.trader.check_stop_loss_take_profit(current_price, timestamp)
            if trade_result:
                self.monitor.log_trade(
                    timestamp=timestamp,
                    action='SELL',
                    price=trade_result['price'],
                    quantity=trade_result['quantity'],
                    amount=trade_result['amount'],
                    reason=trade_result['reason'],
                    pnl=trade_result['pnl'],
                    pnl_percent=trade_result['pnl_percent'],
                    capital_after=self.trader.cash
                )
                self.risk_manager.update_after_trade(trade_result)
                return

        # 策略分析
        analysis = self.strategy.analyze(self.history_data, current_index)

        signal = analysis['signal']
        signal_strength = analysis['signal_strength']
        reasons = analysis['reasons']
        risk_level = analysis['risk_level']

        self.logger.info(f"信号: {signal.value}, 强度: {signal_strength.value if signal_strength else 'N/A'}")
        self.logger.info(f"原因: {reasons}")
        self.logger.info(f"风险等级: {risk_level}")

        # 风险检查
        passed_risk_check, risk_reason = self.risk_manager.check_risk_before_trade(
            current_capital=self.trader.get_total_equity(current_price),
            initial_capital=self.trader.initial_capital,
            recent_trades=self.trader.trade_history,
            current_price=current_price,
            price_history=self.history_data.iloc[max(0, current_index-10):current_index],
            timestamp=timestamp
        )

        if not passed_risk_check:
            self.logger.warning(f"风险检查不通过: {risk_reason}")
            if self.trader.position.is_open:
                self.logger.warning(f"强制平仓")
                trade_result = self.trader.execute_sell(current_price, timestamp, f"风险平仓: {risk_reason}")
                if trade_result:
                    self.monitor.log_trade(
                        timestamp=timestamp,
                        action='SELL',
                        price=trade_result['price'],
                        quantity=trade_result['quantity'],
                        amount=trade_result['amount'],
                        reason=trade_result['reason'],
                        pnl=trade_result['pnl'],
                        pnl_percent=trade_result['pnl_percent'],
                        capital_after=self.trader.cash
                    )
            return

        # 执行交易信号
        if signal == SignalType.BUY and not self.trader.position.is_open:
            strength_str = signal_strength.value if signal_strength else "中"
            trade_result = self.trader.execute_buy(current_price, timestamp, strength_str)
            if trade_result:
                self.monitor.log_trade(
                    timestamp=timestamp,
                    action='BUY',
                    price=trade_result['price'],
                    quantity=trade_result['quantity'],
                    amount=trade_result['amount'],
                    reason=reasons,
                    capital_after=self.trader.cash
                )

        elif signal == SignalType.SELL and self.trader.position.is_open:
            trade_result = self.trader.execute_sell(current_price, timestamp, reasons)
            if trade_result:
                self.monitor.log_trade(
                    timestamp=timestamp,
                    action='SELL',
                    price=trade_result['price'],
                    quantity=trade_result['quantity'],
                    amount=trade_result['amount'],
                    reason=trade_result['reason'],
                    pnl=trade_result['pnl'],
                    pnl_percent=trade_result['pnl_percent'],
                    capital_after=self.trader.cash
                )
                self.risk_manager.update_after_trade(trade_result)

        # 显示当前状态
        total_equity = self.trader.get_total_equity(current_price)
        self.logger.info(f"当前资金: {self.trader.cash:.2f} USDT")
        self.logger.info(f"总权益: {total_equity:.2f} USDT")
        self.logger.info(f"持仓: {'是' if self.trader.position.is_open else '否'}")

    def _hourly_report(self):
        """每小时报告"""
        if self.start_time is None:
            return

        elapsed = datetime.now() - self.start_time
        hours_elapsed = elapsed.total_seconds() / 3600

        # 检查是否到了新的一小时
        if self.last_hourly_report is None or (datetime.now() - self.last_hourly_report).total_seconds() >= 3600:
            self.last_hourly_report = datetime.now()

            stats = self.trader.get_performance_stats()

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"第 {int(hours_elapsed)} 小时报告")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"已运行: {hours_elapsed:.1f} / {self.total_hours} 小时")
            self.logger.info(f"总交易次数: {stats['total_trades']}")
            self.logger.info(f"盈利次数: {stats['winning_trades']}")
            self.logger.info(f"亏损次数: {stats['losing_trades']}")
            self.logger.info(f"当前资金: {stats['final_capital']:.2f} USDT")
            self.logger.info(f"总收益: {stats['total_return']:.2f}%")
            self.logger.info(f"最大回撤: {stats['max_drawdown']:.2f}%")
            self.logger.info(f"{'='*60}\n")

    def run(self):
        """开始实时模拟交易"""
        self.logger.info("="*60)
        self.logger.info("实时交易模拟器启动")
        self.logger.info(f"交易对: {self.symbol}")
        self.logger.info(f"时间周期: {self.timeframe}")
        self.logger.info(f"运行时长: {self.total_hours} 小时")
        self.logger.info(f"每 {self.interval_seconds} 秒更新一次")
        self.logger.info("="*60)

        # 获取初始历史数据
        self._fetch_initial_history()

        # 设置运行时间
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=self.total_hours)
        self.is_running = True

        self.logger.info(f"\n开始时间: {self.start_time}")
        self.logger.info(f"预计结束时间: {self.end_time}")
        self.logger.info(f"需要运行: {self.total_hours} 小时 ({self.total_hours / 24:.1f} 天)\n")

        try:
            while self.is_running and datetime.now() < self.end_time:
                iteration_start = time.time()

                # 获取最新K线
                new_bar = self._fetch_latest_bar()

                if not new_bar.empty:
                    # 更新历史数据
                    self._update_history_data(new_bar)

                    # 执行交易逻辑
                    self._process_trading_logic()

                    # 每小时报告
                    self._hourly_report()

                self.total_iterations += 1

                # 计算下次更新时间
                elapsed = time.time() - iteration_start
                sleep_time = max(0, self.interval_seconds - elapsed)

                if sleep_time > 0:
                    self.logger.debug(f"等待 {sleep_time:.1f} 秒到下一次更新...")
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("\n收到中断信号，正在停止...")
        except Exception as e:
            self.logger.error(f"运行出错: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _shutdown(self):
        """关闭并生成最终报告"""
        self.is_running = False

        self.logger.info("\n正在生成最终报告...")

        # 如果有持仓，平仓
        if self.trader.position.is_open and len(self.history_data) > 0:
            final_price = self.history_data.iloc[-1]['close']
            final_timestamp = self.history_data.iloc[-1]['timestamp']
            trade_result = self.trader.execute_sell(final_price, final_timestamp, "模拟结束平仓")
            if trade_result:
                self.monitor.log_trade(
                    timestamp=final_timestamp,
                    action='SELL',
                    price=trade_result['price'],
                    quantity=trade_result['quantity'],
                    amount=trade_result['amount'],
                    reason=trade_result['reason'],
                    pnl=trade_result['pnl'],
                    pnl_percent=trade_result['pnl_percent'],
                    capital_after=self.trader.cash
                )

        # 生成报告
        actual_duration = (datetime.now() - self.start_time).total_seconds() / 3600

        report = self.monitor.generate_summary_report(
            initial_capital=self.trader.initial_capital,
            final_capital=self.trader.cash,
            trade_history=self.trader.trade_history,
            equity_curve=self.trader.equity_curve,
            duration_hours=actual_duration
        )

        # 绘制图表
        if self.config.get('plot_charts', True):
            self.monitor.plot_equity_curve(self.trader.equity_curve)

        self.logger.info("\n实时模拟交易已结束")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实时交易模拟器 - 从现在开始往后运行72小时')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--hours',
        type=int,
        help='运行小时数（覆盖配置文件）'
    )

    args = parser.parse_args()

    # 运行实时模拟
    simulator = LiveTradingSimulator(config_path=args.config)

    # 如果命令行指定了小时数，覆盖配置
    if args.hours:
        simulator.total_hours = args.hours
        simulator.logger.info(f"使用命令行指定的运行时长: {args.hours} 小时")

    # 设置信号处理（Ctrl+C优雅退出）
    def signal_handler(sig, frame):
        simulator.logger.info("\n收到停止信号...")
        simulator.is_running = False

    signal.signal(signal.SIGINT, signal_handler)

    simulator.run()


if __name__ == "__main__":
    main()
