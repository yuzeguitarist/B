"""
数据获取模块 - 通过币安API获取历史K线数据
"""
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """币安数据获取器"""

    def __init__(self, api_key: str = "", api_secret: str = ""):
        """
        初始化币安交易所连接
        Args:
            api_key: API密钥（获取历史数据可以不需要）
            api_secret: API密钥
        """
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # 启用速率限制
            'options': {
                'defaultType': 'spot',  # 现货交易
            }
        })

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        hours: int = 72,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        获取OHLCV数据

        Args:
            symbol: 交易对，如'BTC/USDT'
            timeframe: 时间周期，默认'1m'
            hours: 获取小时数，默认72小时
            end_time: 结束时间，默认为当前时间

        Returns:
            DataFrame包含：timestamp, open, high, low, close, volume
        """
        logger.info(f"开始获取 {symbol} {hours}小时的{timeframe}数据")

        if end_time is None:
            end_time = datetime.now()

        start_time = end_time - timedelta(hours=hours)

        # 转换为毫秒时间戳
        since = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        all_ohlcv = []
        current_since = since

        # 分批获取数据（避免API限制）
        batch_size = 1000  # 每次最多获取1000条
        retry_count = 0
        max_retries = 5

        while current_since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=batch_size
                )

                if not ohlcv:
                    logger.warning("未获取到数据，跳出循环")
                    break

                all_ohlcv.extend(ohlcv)

                # 更新下一批的起始时间
                current_since = ohlcv[-1][0] + 1

                logger.info(f"已获取 {len(all_ohlcv)} 条数据，继续获取...")

                # 如果获取的数据少于batch_size，说明已经到达最新数据
                if len(ohlcv) < batch_size:
                    break

                # 避免触发速率限制
                time.sleep(self.exchange.rateLimit / 1000)
                retry_count = 0  # 重置重试计数

            except ccxt.RateLimitExceeded as e:
                logger.warning(f"触发速率限制，等待后重试: {e}")
                time.sleep(5)
                continue

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"获取数据失败，已重试{max_retries}次: {e}")
                    raise
                logger.warning(f"获取数据出错，重试 {retry_count}/{max_retries}: {e}")
                time.sleep(2 ** retry_count)  # 指数退避

        # 过滤超出时间范围的数据
        all_ohlcv = [candle for candle in all_ohlcv if since <= candle[0] <= end_ts]

        logger.info(f"数据获取完成，总共 {len(all_ohlcv)} 条记录")

        # 转换为DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def get_latest_price(self, symbol: str) -> float:
        """获取最新价格"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"获取最新价格失败: {e}")
            raise

    def get_exchange_info(self, symbol: str) -> dict:
        """获取交易对信息（最小交易量等）"""
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                return markets[symbol]
            else:
                raise ValueError(f"交易对 {symbol} 不存在")
        except Exception as e:
            logger.error(f"获取交易对信息失败: {e}")
            raise


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    fetcher = BinanceDataFetcher()

    # 获取BTC/USDT最近72小时的1分钟数据
    df = fetcher.fetch_ohlcv('BTC/USDT', timeframe='1m', hours=1)  # 测试用1小时
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
