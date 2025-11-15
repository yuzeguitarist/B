"""
数据清洗模块 - 处理缺失值、异常值、时间戳连续性
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗器"""

    def __init__(self, timeframe: str = '1m'):
        """
        初始化
        Args:
            timeframe: 时间周期，用于计算期望的时间间隔
        """
        self.timeframe = timeframe
        self.interval_seconds = self._parse_timeframe(timeframe)

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

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的数据清洗流程

        Args:
            df: 原始数据

        Returns:
            清洗后的数据
        """
        logger.info(f"开始数据清洗，原始数据: {len(df)} 条")

        # 1. 检查必要列
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")

        # 2. 删除完全重复的行
        df = self._remove_duplicates(df)

        # 3. 检查时间戳连续性
        df = self._check_timestamp_continuity(df)

        # 4. 处理缺失值
        df = self._handle_missing_values(df)

        # 5. 异常值检测与处理
        df = self._detect_and_handle_outliers(df)

        # 6. 数据验证
        df = self._validate_data(df)

        # 7. 重置索引
        df = df.reset_index(drop=True)

        logger.info(f"数据清洗完成，清洗后数据: {len(df)} 条")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除重复数据"""
        original_len = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        removed = original_len - len(df)
        if removed > 0:
            logger.warning(f"删除了 {removed} 条重复记录")
        return df

    def _check_timestamp_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """检查时间戳连续性，填补缺失的时间点"""
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 计算时间差
        time_diffs = df['timestamp'].diff()

        # 期望的时间间隔
        expected_interval = pd.Timedelta(seconds=self.interval_seconds)

        # 查找缺失的时间点
        gaps = time_diffs[time_diffs > expected_interval * 1.5]

        if len(gaps) > 0:
            logger.warning(f"发现 {len(gaps)} 个时间间隔异常")

            # 创建完整的时间索引
            full_range = pd.date_range(
                start=df['timestamp'].min(),
                end=df['timestamp'].max(),
                freq=f'{self.interval_seconds}S'
            )

            # 重新索引并填充
            df = df.set_index('timestamp')
            df = df.reindex(full_range)
            df.index.name = 'timestamp'
            df = df.reset_index()

            logger.info(f"填补时间间隔后，数据量: {len(df)}")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 统计缺失值
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"发现缺失值:\n{missing_counts[missing_counts > 0]}")

            # 对于价格数据，使用前向填充
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].fillna(method='ffill')

            # 如果还有缺失（开头部分），使用后向填充
            df[price_cols] = df[price_cols].fillna(method='bfill')

            # 成交量缺失填0
            df['volume'] = df['volume'].fillna(0)

            # 检查是否还有缺失
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"填充后仍有 {remaining_missing} 个缺失值，将删除这些行")
                df = df.dropna()

        return df

    def _detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测并处理异常值"""
        df = df.copy()

        # 1. 价格突刺检测（相邻K线价格变化超过20%视为异常）
        for col in ['open', 'high', 'low', 'close']:
            pct_change = df[col].pct_change().abs()
            outliers = pct_change > 0.20  # 20%阈值

            if outliers.sum() > 0:
                logger.warning(f"{col} 列发现 {outliers.sum()} 个价格突刺异常")

                # 使用前后平均值替换异常值
                for idx in df[outliers].index:
                    if idx > 0 and idx < len(df) - 1:
                        df.loc[idx, col] = (df.loc[idx-1, col] + df.loc[idx+1, col]) / 2
                    elif idx == 0 and len(df) > 1:
                        df.loc[idx, col] = df.loc[idx+1, col]
                    elif idx == len(df) - 1 and len(df) > 1:
                        df.loc[idx, col] = df.loc[idx-1, col]

        # 2. 成交量异常检测（使用IQR方法）
        Q1 = df['volume'].quantile(0.25)
        Q3 = df['volume'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        volume_outliers = (df['volume'] < lower_bound) | (df['volume'] > upper_bound)
        if volume_outliers.sum() > 0:
            logger.warning(f"成交量发现 {volume_outliers.sum()} 个异常值")
            # 对于成交量异常，使用中位数替换
            median_volume = df['volume'].median()
            df.loc[volume_outliers, 'volume'] = median_volume

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据合法性验证"""
        invalid_rows = []

        # 检查 high >= low
        invalid = df['high'] < df['low']
        if invalid.sum() > 0:
            logger.warning(f"发现 {invalid.sum()} 条数据 high < low")
            invalid_rows.extend(df[invalid].index.tolist())

        # 检查 high >= open, close
        invalid = (df['high'] < df['open']) | (df['high'] < df['close'])
        if invalid.sum() > 0:
            logger.warning(f"发现 {invalid.sum()} 条数据 high < open/close")
            invalid_rows.extend(df[invalid].index.tolist())

        # 检查 low <= open, close
        invalid = (df['low'] > df['open']) | (df['low'] > df['close'])
        if invalid.sum() > 0:
            logger.warning(f"发现 {invalid.sum()} 条数据 low > open/close")
            invalid_rows.extend(df[invalid].index.tolist())

        # 检查价格和成交量 > 0
        invalid = (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)
        if invalid.sum() > 0:
            logger.warning(f"发现 {invalid.sum()} 条数据价格 <= 0")
            invalid_rows.extend(df[invalid].index.tolist())

        # 删除无效数据
        if invalid_rows:
            invalid_rows = list(set(invalid_rows))
            logger.warning(f"总共删除 {len(invalid_rows)} 条无效数据")
            df = df.drop(invalid_rows)

        return df

    def get_data_quality_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
        """生成数据质量报告"""
        report = {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'removed_rows': len(original_df) - len(cleaned_df),
            'removal_rate': (len(original_df) - len(cleaned_df)) / len(original_df) * 100,
            'time_range': {
                'start': cleaned_df['timestamp'].min(),
                'end': cleaned_df['timestamp'].max(),
                'duration_hours': (cleaned_df['timestamp'].max() - cleaned_df['timestamp'].min()).total_seconds() / 3600
            },
            'price_stats': {
                'min': cleaned_df['low'].min(),
                'max': cleaned_df['high'].max(),
                'mean': cleaned_df['close'].mean(),
            },
            'volume_stats': {
                'total': cleaned_df['volume'].sum(),
                'mean': cleaned_df['volume'].mean(),
                'median': cleaned_df['volume'].median(),
            }
        }
        return report


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.rand(100) * 1000
    })

    cleaner = DataCleaner(timeframe='1m')
    cleaned_df = cleaner.clean_data(test_df)
    report = cleaner.get_data_quality_report(test_df, cleaned_df)

    print("数据质量报告:")
    for key, value in report.items():
        print(f"{key}: {value}")
