# 量化交易系统升级功能说明

## 概述

本次升级为币安量化回测系统添加了5个核心功能，涵盖资金面监控、情绪面分析和高级风控策略。

## 新增功能

### 1. 资金费率监控 (Funding Rate Monitor)

**文件**: `src/funding_rate.py`

**功能描述**:
- 从币安期货市场实时获取永续合约的资金费率
- 当资金费率 > 0.1% 时，警告市场超买（多头支付空头，看多情绪高涨）
- 当资金费率 < -0.1% 时，提示市场超卖（空头支付多头，看空情绪高涨）
- 自动集成到策略评分系统，评分权重：-15 到 15

**配置参数** (config.yaml):
```yaml
strategy:
  funding_rate_overbought: 0.001   # 0.1% - 超买阈值
  funding_rate_oversold: -0.001    # -0.1% - 超卖阈值
  funding_rate_extreme: 0.002      # 0.2% - 极端阈值
  funding_rate_weight: 15          # 评分权重
```

**使用示例**:
```python
from src.funding_rate import FundingRateMonitor

monitor = FundingRateMonitor()
funding_rate = monitor.get_funding_rate('BTC/USDT')
analysis = monitor.analyze_funding_rate(funding_rate)
print(f"资金费率: {analysis['funding_rate_percent']:.4f}%")
print(f"市场情绪: {analysis['sentiment']}")
```

---

### 2. 波动率动态仓位 (Volatility-Based Position Sizing)

**文件**: `src/volatility_position.py`

**功能描述**:
- 根据历史波动率动态调整仓位大小
- 使用公式：仓位 = (可接受亏损 / 波动率)
- 高波动时减少仓位，低波动时增加仓位
- 替换固定仓位比例，提高资金使用效率

**核心特性**:
- 支持标准差(std)和ATR两种波动率计算方法
- 基于固定风险和目标波动率的双重仓位计算
- 根据信号强度自动调整仓位倍数
- 动态计算止损止盈价格

**配置参数** (config.yaml):
```yaml
trading:
  volatility_position:
    use_volatility_position: true      # 是否启用
    volatility_period: 20              # 波动率计算周期
    volatility_method: std             # std或atr
    max_risk_per_trade: 0.02           # 单次交易最大风险2%
    min_position_percent: 0.1          # 最小仓位10%
    max_position_percent: 0.95         # 最大仓位95%
    volatility_target: 0.02            # 目标波动率2%
```

**使用示例**:
```python
from src.volatility_position import VolatilityPositionSizer

sizer = VolatilityPositionSizer()
volatility = sizer.calculate_volatility(df, index)
pos_pct, buy_amt, desc = sizer.calculate_position_size(
    current_capital=10000,
    current_price=50000,
    volatility=volatility,
    signal_strength='强'
)
print(f"建议仓位: {pos_pct*100:.1f}%")
```

---

### 3. 移动止损 (Trailing Stop Loss)

**文件**: `src/trailing_stop.py`

**功能描述**:
- 盈利后自动激活移动止损机制
- 止损价随价格上涨自动上移，保护利润
- 默认：盈利2%后激活，从最高点回撤3%触发止损
- 支持多级移动止损策略

**核心特性**:
- 盈利达到阈值后自动激活
- 止损价只能上移，不能下移
- 实时跟踪最高价格
- 提供详细的状态信息

**配置参数** (config.yaml):
```yaml
trading:
  trailing_stop:
    use_trailing_stop: true            # 是否启用
    trailing_stop_activation: 0.02     # 盈利2%后激活
    trailing_stop_distance: 0.03       # 回撤3%触发止损
    trailing_stop_step: 0.01           # 止损价上移最小步长1%
```

**使用示例**:
```python
from src.trailing_stop import TrailingStopManager

trailing = TrailingStopManager()
trailing.initialize(entry_price=50000, initial_stop_loss=49000)

# 每次价格更新时调用
stop_price, triggered, message = trailing.update(
    current_price=51500,
    entry_price=50000
)
if triggered:
    print(f"触发移动止损: {message}")
```

---

### 4. 分批止盈 (Partial Profit Taking)

**文件**: `src/partial_profit.py`

**功能描述**:
- 在不同盈利水平自动分批卖出，锁定利润
- 默认策略：
  - 盈利30%时卖出1/3仓位
  - 盈利60%时再卖出1/3仓位（剩余的一半）
  - 剩余1/3使用移动止损
- 支持动态调整止盈目标（根据波动率）

**核心特性**:
- 灵活的分批止盈级别配置
- 基于剩余仓位计算卖出数量
- 实时跟踪已实现利润
- 支持波动率自适应调整

**配置参数** (config.yaml):
```yaml
trading:
  partial_profit:
    use_partial_profit: true           # 是否启用
    profit_levels:                     # 分批止盈级别
      - [0.30, 0.33]                   # 盈利30%时卖出33%
      - [0.60, 0.50]                   # 盈利60%时卖出50%（基于剩余仓位）
```

**使用示例**:
```python
from src.partial_profit import PartialProfitManager

partial = PartialProfitManager()
partial.initialize(quantity=1.0)  # 初始1个BTC

# 价格更新时检查
level, sell_qty, message = partial.check_and_execute(
    current_price=65000,
    entry_price=50000
)
if level is not None:
    print(f"触发止盈: {message}")
    print(f"卖出数量: {sell_qty}")
```

---

### 5. 多空比监控 (Long/Short Ratio Monitor)

**文件**: `src/long_short_ratio.py`

**功能描述**:
- 监控市场多头/空头账户比例
- 获取持仓量(Open Interest)数据
- 当多空比 > 3:1 时，警告多头拥挤，警惕反转
- 当多空比 < 1:3 时，提示空头拥挤，关注反弹
- 自动集成到策略评分系统

**核心特性**:
- 账户多空比（散户情绪）
- 大户多空比（机构情绪）
- 持仓量监控
- 拥挤度分析
- 逆向交易信号

**配置参数** (config.yaml):
```yaml
strategy:
  extreme_long_ratio: 3.0          # 多空比>3:1警告
  extreme_short_ratio: 0.33        # 多空比<1:3警告
  crowded_threshold: 2.5           # 拥挤阈值
  long_short_ratio_weight: 10      # 评分权重
```

**使用示例**:
```python
from src.long_short_ratio import LongShortRatioMonitor

monitor = LongShortRatioMonitor()
ls_ratio = monitor.get_long_short_ratio('BTC/USDT')
analysis = monitor.analyze_long_short_ratio(ls_ratio)
print(f"多空比: {analysis['ls_ratio']:.2f}:1")
print(f"市场情绪: {analysis['sentiment']}")
print(f"是否拥挤: {analysis['crowded']}")
```

---

## 集成说明

### 策略评分系统集成

所有新功能已自动集成到 `src/strategy.py` 的综合评分系统中：

- **资金费率**: -15 到 15 分
- **多空比**: -10 到 10 分
- **原有技术指标**: RSI、KDJ、MACD、布林带、成交量、趋势等

总评分范围扩展为约 -130 到 130 分（之前为 -100 到 100）

### 使用方式

1. **回测系统自动使用**：
   - 策略评分已自动包含资金费率和多空比
   - 无需修改现有回测代码

2. **手动使用各模块**：
   ```python
   # 资金费率
   from src.funding_rate import FundingRateMonitor
   funding_monitor = FundingRateMonitor(config)

   # 多空比
   from src.long_short_ratio import LongShortRatioMonitor
   ls_monitor = LongShortRatioMonitor(config)

   # 波动率仓位
   from src.volatility_position import VolatilityPositionSizer
   position_sizer = VolatilityPositionSizer(config)

   # 移动止损
   from src.trailing_stop import TrailingStopManager
   trailing = TrailingStopManager(config)

   # 分批止盈
   from src.partial_profit import PartialProfitManager
   partial = PartialProfitManager(config)
   ```

---

## 配置建议

### 保守策略
```yaml
trading:
  volatility_position:
    max_risk_per_trade: 0.01    # 降低单次风险到1%
  trailing_stop:
    trailing_stop_activation: 0.03  # 盈利3%后激活
    trailing_stop_distance: 0.02    # 回撤2%触发
  partial_profit:
    profit_levels:
      - [0.20, 0.50]    # 盈利20%卖一半
      - [0.40, 0.50]    # 盈利40%再卖一半
```

### 激进策略
```yaml
trading:
  volatility_position:
    max_risk_per_trade: 0.03    # 提高单次风险到3%
  trailing_stop:
    trailing_stop_activation: 0.01  # 盈利1%即激活
    trailing_stop_distance: 0.05    # 回撤5%触发
  partial_profit:
    profit_levels:
      - [0.50, 0.33]    # 盈利50%卖1/3
      - [1.00, 0.50]    # 盈利100%再卖一半
```

---

## 测试说明

每个模块都包含独立的测试代码，可以单独运行：

```bash
# 测试资金费率监控
python -m src.funding_rate

# 测试多空比监控
python -m src.long_short_ratio

# 测试波动率动态仓位
python -m src.volatility_position

# 测试移动止损
python -m src.trailing_stop

# 测试分批止盈
python -m src.partial_profit
```

---

## 注意事项

1. **API限制**：
   - 资金费率和多空比需要访问币安API
   - 建议设置合理的缓存时间（默认5分钟）
   - 避免频繁请求导致限流

2. **回测vs实盘**：
   - 波动率、移动止损、分批止盈可用于回测
   - 资金费率和多空比数据在回测时需要历史数据支持
   - 当前版本在回测时这些数据可能不可用，会使用默认值

3. **参数调优**：
   - 建议先在回测中测试不同参数组合
   - 根据币种特性和市场环境调整参数
   - 定期review和优化策略参数

4. **风险控制**：
   - 新功能增加了策略复杂度
   - 建议先小仓位测试
   - 密切监控实盘表现

---

## 未来扩展

根据用户需求，可以继续添加：
- 链上数据监控（交易所流向、大户地址）
- 恐惧贪婪指数集成
- 多币种组合管理
- 机器学习优化参数

---

## 技术支持

如有问题或建议，请提交Issue或联系开发团队。
