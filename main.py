"""
币安自动交易策略回测系统 - 主程序
"""
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_fetcher import BinanceDataFetcher
from data_cleaner import DataCleaner
from indicators import TechnicalIndicators
from strategy import TradingStrategy, SignalType
from trader import TradingSimulator
from risk_manager import RiskManager
from monitor import BacktestMonitor


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化回测引擎

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设置日志
        self._setup_logging()

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

        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        """设置日志"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
            ]
        )

    def run(self):
        """运行回测"""
        self.logger.info("="*60)
        self.logger.info("币安自动交易策略回测系统启动")
        self.logger.info("="*60)

        try:
            # 1. 获取数据
            self.logger.info("步骤1: 获取历史数据...")
            df = self._fetch_data()

            # 2. 清洗数据
            self.logger.info("步骤2: 清洗数据...")
            df = self._clean_data(df)

            # 3. 计算技术指标
            self.logger.info("步骤3: 计算技术指标...")
            df = self._calculate_indicators(df)

            # 4. 执行回测
            self.logger.info("步骤4: 执行回测...")
            self._backtest(df)

            # 5. 生成报告
            self.logger.info("步骤5: 生成报告...")
            self._generate_report(df)

            self.logger.info("\n回测完成！")

        except Exception as e:
            self.logger.error(f"回测失败: {e}", exc_info=True)
            raise

    def _fetch_data(self) -> pd.DataFrame:
        """获取数据"""
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        hours = self.config['hours']

        # 如果配置了使用本地数据
        if self.config.get('use_local_data', False):
            data_file = self.config.get('local_data_file')
            self.logger.info(f"从本地文件加载数据: {data_file}")
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df

        # 从币安API获取
        df = self.data_fetcher.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            hours=hours
        )

        # 保存原始数据
        if self.config.get('save_raw_data', True):
            raw_data_file = f"data/raw_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(raw_data_file, index=False)
            self.logger.info(f"原始数据已保存: {raw_data_file}")

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        original_df = df.copy()

        cleaned_df = self.data_cleaner.clean_data(df)

        # 生成数据质量报告
        quality_report = self.data_cleaner.get_data_quality_report(original_df, cleaned_df)
        self.logger.info(f"数据质量报告: 原始{quality_report['original_rows']}条, "
                        f"清洗后{quality_report['cleaned_rows']}条, "
                        f"删除{quality_report['removed_rows']}条")

        return cleaned_df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        indicator_calculator = TechnicalIndicators(df)
        indicator_config = self.config.get('indicators', {})
        df = indicator_calculator.calculate_all_indicators(indicator_config)

        self.logger.info(f"技术指标计算完成，共{len(df.columns)}列")

        return df

    def _backtest(self, df: pd.DataFrame):
        """执行回测"""
        total_bars = len(df)
        self.logger.info(f"开始回测，总K线数: {total_bars}")

        # 从第100根K线开始（需要足够的历史数据计算指标）
        start_index = 100
        hour_counter = 0
        last_hour_index = start_index

        for i in range(start_index, total_bars):
            current_bar = df.iloc[i]
            timestamp = current_bar['timestamp']
            current_price = current_bar['close']

            # 每小时统计
            if i - last_hour_index >= 60:  # 假设1分钟K线，60根=1小时
                hour_counter += 1
                last_hour_index = i

                stats = self.trader.get_performance_stats()
                self.monitor.log_hourly_stats(
                    hour=hour_counter,
                    timestamp=timestamp,
                    trades_count=stats['total_trades'],
                    winning_trades=stats['winning_trades'],
                    losing_trades=stats['losing_trades'],
                    total_pnl=stats['final_capital'] - stats['initial_capital'],
                    capital=stats['final_capital'],
                    max_drawdown=stats['max_drawdown']
                )

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
                    continue

            # 策略分析
            analysis = self.strategy.analyze(df, i)

            signal = analysis['signal']
            signal_strength = analysis['signal_strength']
            reasons = analysis['reasons']
            risk_level = analysis['risk_level']

            # 风险检查
            passed_risk_check, risk_reason = self.risk_manager.check_risk_before_trade(
                current_capital=self.trader.get_total_equity(current_price),
                initial_capital=self.trader.initial_capital,
                recent_trades=self.trader.trade_history,
                current_price=current_price,
                price_history=df.iloc[max(0, i-10):i],
                timestamp=timestamp
            )

            if not passed_risk_check:
                # 如果有持仓且风险检查不通过，考虑平仓
                if self.trader.position.is_open:
                    self.logger.warning(f"风险检查不通过，平仓: {risk_reason}")
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
                continue

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

            # 进度显示
            if i % 500 == 0:
                progress = (i - start_index) / (total_bars - start_index) * 100
                self.logger.info(f"回测进度: {progress:.1f}% ({i}/{total_bars})")

        # 如果最后还有持仓，平仓
        if self.trader.position.is_open:
            final_price = df.iloc[-1]['close']
            final_timestamp = df.iloc[-1]['timestamp']
            trade_result = self.trader.execute_sell(final_price, final_timestamp, "回测结束平仓")
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

    def _generate_report(self, df: pd.DataFrame):
        """生成报告"""
        duration_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600

        report = self.monitor.generate_summary_report(
            initial_capital=self.trader.initial_capital,
            final_capital=self.trader.cash,
            trade_history=self.trader.trade_history,
            equity_curve=self.trader.equity_curve,
            duration_hours=duration_hours
        )

        # 绘制图表
        if self.config.get('plot_charts', True):
            self.monitor.plot_equity_curve(self.trader.equity_curve)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='币安自动交易策略回测系统')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='配置文件路径'
    )

    args = parser.parse_args()

    # 运行回测
    engine = BacktestEngine(config_path=args.config)
    engine.run()


if __name__ == "__main__":
    main()
