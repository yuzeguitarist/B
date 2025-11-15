# 快速开始指南

## 5分钟快速上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行第一个回测

使用默认配置（BTC/USDT，72小时，1000 USDT初始资金）：

```bash
python main.py
```

### 3. 查看结果

回测完成后，查看 `results/` 目录：

```bash
ls -lh results/
```

你会看到：
- `backtest_*.log` - 详细日志
- `report_*.json` - 总结报告
- `trades_*.csv` - 所有交易记录
- `equity_*.csv` - 权益曲线数据
- `equity_curve_*.png` - 可视化图表

### 4. 查看报告示例

```bash
# 查看最新的JSON报告
cat results/report_*.json | python -m json.tool
```

报告包含：
```json
{
  "total_return_percent": 5.2,
  "total_trades": 15,
  "win_rate": 60.0,
  "profit_factor": 2.1,
  "max_drawdown": 3.5,
  "sharpe_ratio": 1.8
}
```

## 自定义配置示例

### 示例1：更保守的策略

编辑 `config/config.yaml`：

```yaml
trading:
  initial_capital: 1000
  stop_loss_percent: 0.015    # 1.5%止损（更严格）
  take_profit_percent: 0.03   # 3%止盈（更保守）
  max_position_percent: 0.7   # 70%最大仓位

strategy:
  buy_score_threshold: 70     # 更高的买入阈值
  sell_score_threshold: -70   # 更高的卖出阈值
```

### 示例2：更激进的策略

```yaml
trading:
  initial_capital: 1000
  stop_loss_percent: 0.03     # 3%止损（更宽松）
  take_profit_percent: 0.08   # 8%止盈（更激进）
  max_position_percent: 0.95  # 95%最大仓位

strategy:
  buy_score_threshold: 50     # 更低的买入阈值
  sell_score_threshold: -50   # 更低的卖出阈值
```

### 示例3：短期交易

```yaml
trading:
  max_holding_minutes: 60     # 最多持仓1小时

indicators:
  ema_periods: [5, 15, 30]    # 更短周期的均线
  rsi_period: 7               # 更灵敏的RSI
```

## 测试不同交易对

修改配置文件中的交易对：

```yaml
# 以太坊
symbol: "ETH/USDT"

# 币安币
symbol: "BNB/USDT"

# 瑞波币
symbol: "XRP/USDT"
```

## 使用本地数据（避免API调用）

1. 第一次运行时保存数据：

```yaml
save_raw_data: true
```

2. 后续使用本地数据：

```yaml
use_local_data: true
local_data_file: "data/raw_BTC_USDT_20241115_123456.csv"
```

## 常用命令

```bash
# 查看最新的交易日志
tail -f results/backtest_*.log

# 统计交易次数
grep "买入成功\|卖出成功" results/backtest_*.log | wc -l

# 查看盈利交易
grep "卖出成功" results/backtest_*.log | grep "盈亏=[0-9]"

# 查看亏损交易
grep "卖出成功" results/backtest_*.log | grep "盈亏=-"
```

## 性能优化建议

### 如果回测太慢：

1. **使用本地数据**（最有效）
```yaml
use_local_data: true
```

2. **减少回测时长**
```yaml
hours: 24  # 从72小时减少到24小时
```

3. **使用更大的时间周期**
```yaml
timeframe: "5m"  # 从1分钟改为5分钟
```

### 如果内存不足：

1. 减少回测时长
2. 关闭图表绘制：
```yaml
plot_charts: false
```

## 下一步

1. 阅读完整的 [README.md](README.md) 了解详细功能
2. 调整策略参数进行优化
3. 尝试不同的交易对和时间周期
4. 分析失败交易，改进策略

## 技巧和窍门

### 快速测试策略

使用短时间回测快速验证策略：

```yaml
hours: 6  # 只测试6小时
```

### 对比不同策略

1. 复制配置文件：
```bash
cp config/config.yaml config/config_aggressive.yaml
cp config/config.yaml config/config_conservative.yaml
```

2. 分别运行：
```bash
python main.py --config config/config_aggressive.yaml
python main.py --config config/config_conservative.yaml
```

3. 对比结果

### 记录最佳参数

在 `config/` 目录创建不同的配置文件：
- `config_btc_best.yaml` - BTC最佳参数
- `config_eth_best.yaml` - ETH最佳参数
- `config_scalping.yaml` - 短线策略
- `config_swing.yaml` - 波段策略

## 故障排除

### 问题：无法连接币安API

解决：
1. 检查网络连接
2. 使用本地数据（`use_local_data: true`）
3. 检查API密钥（获取历史数据通常不需要）

### 问题：没有交易发生

可能原因：
1. 策略阈值太高 - 降低 `buy_score_threshold`
2. 风险控制太严 - 调整 `risk` 配置
3. 数据时间段市场波动小 - 换个时间段

### 问题：回测亏损严重

建议：
1. 检查止损设置是否合理
2. 降低仓位比例
3. 提高买入/卖出信号阈值
4. 调整技术指标参数

## 获取帮助

遇到问题？
1. 查看 `results/backtest_*.log` 日志文件
2. 检查配置文件语法
3. 阅读 [README.md](README.md) 中的常见问题
4. 提交 Issue

---

祝回测顺利！
