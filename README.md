# AI策略系统

基于人工智能的量化交易策略系统，集成多种数据源和AI模型进行智能决策。

## 功能特性

- 🤖 **AI驱动决策**: 集成GLM、OpenAI、百度等多种AI模型
- 📊 **多数据源支持**: TQSDK、通达信、AkShare等
- 🔄 **智能回测引擎**: 完整的策略回测和性能分析
- ⚡ **实时监控**: 性能监控和参数优化
- 🔒 **安全配置**: 环境变量管理，敏感信息不上传

## 快速开始

### 1. 环境配置

复制环境变量模板并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的配置：

```bash
# GLM API配置
GLM_API_KEY=your-glm-api-key

# TQSDK配置
TQSDK_ACCOUNT=your-account
TQSDK_PASSWORD=your-password

# 通达信配置
TDX_PATH=/path/to/tdx/vipdoc
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 测试配置

```bash
python config_manager.py
```

### 4. 运行策略

```bash
python ai_strategy_system.py
```

## 项目结构

```
.
├── ai_agent_strategy.py          # AI代理策略
├── ai_strategy_integration.py    # 策略集成
├── ai_strategy_system.py         # 主策略系统
├── backtest_engine.py            # 回测引擎
├── config_manager.py             # 配置管理器
├── enhanced_ai_strategy.py       # 增强AI策略
├── enhanced_data_source.py       # 增强数据源
├── information_processor.py      # 信息处理器
├── parameter_optimizer.py        # 参数优化器
├── performance_monitor.py        # 性能监控
├── quick_test.py                 # 快速测试
├── real_ai_api.py               # 真实AI API集成
├── signal_decision_system.py    # 信号决策系统
├── .env                         # 环境变量（不上传）
├── .env.example                 # 环境变量模板
└── .gitignore                   # Git忽略文件
```

## 数据源配置

### TQSDK

天勤量化数据源，提供期货、期权等实时行情数据。

```python
TQSDK_ACCOUNT=your-account
TQSDK_PASSWORD=your-password
```

### 通达信

本地通达信数据源，读取历史行情数据。

```python
TDX_PATH=/海王星/vipdoc
TDX_ENABLE=true
```

### AkShare

免费开源的金融数据接口库。

```bash
pip install akshare
```

## AI模型配置

### GLM API

智谱AI的GLM系列模型。

```python
GLM_API_KEY=your-api-key
GLM_MODEL=glm-4
```

### OpenAI (可选)

```python
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4
```

### 百度文心一言 (可选)

```python
BAIDU_API_KEY=your-api-key
BAIDU_SECRET_KEY=your-secret-key
```

## 使用示例

### 基础策略运行

```python
from ai_strategy_system import AIStrategySystem

# 创建策略系统
strategy = AIStrategySystem()

# 运行策略
strategy.run()
```

### 回测

```python
from backtest_engine import BacktestEngine

# 创建回测引擎
engine = BacktestEngine(
    initial_capital=100000,
    commission_rate=0.0003
)

# 运行回测
results = engine.run_backtest(
    strategy=strategy,
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### 参数优化

```python
from parameter_optimizer import ParameterOptimizer

# 创建优化器
optimizer = ParameterOptimizer()

# 优化参数
best_params = optimizer.optimize(
    strategy=strategy,
    param_ranges={
        'momentum_threshold': (0.01, 0.05),
        'rsi_threshold': (30, 70)
    }
)
```

## 安全说明

⚠️ **重要**: 

- `.env` 文件包含敏感信息，已添加到 `.gitignore`，不会上传到Git
- 请勿在代码中硬编码API密钥
- 使用环境变量管理所有敏感配置

## 配置检查

运行配置管理器检查系统配置：

```bash
python config_manager.py
```

输出示例：

```
============================================================
配置摘要
============================================================

【数据源配置】
最佳数据源: tqsdk
可用数据源: tqsdk, akshare
优先级顺序: tqsdk > tdx > akshare

【AI配置】
✓ GLM API: glm-4
✓ OpenAI: gpt-4

【系统配置】
缓存: 启用
日志级别: INFO

【回测配置】
初始资金: 100,000元
手续费率: 0.0300%
============================================================
```

## 开发计划

- [ ] 支持更多数据源
- [ ] 增加更多AI模型
- [ ] 优化策略性能
- [ ] 添加可视化界面
- [ ] 实盘交易接口

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交Issue。
