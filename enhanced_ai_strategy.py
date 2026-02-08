#!/usr/bin/python
"""
增强版AI智能体策略系统
整合TQSDK、本地通达信数据源和GLM-4.7-flash免费模型
基于北京大学光华管理学院前沿研究
"""

import numpy as np
import pandas as pd
import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# 导入自定义模块
try:
    from enhanced_data_source import DataSourceManager, GLMFlashClient
    from parameter_optimizer import ParameterOptimizer
    from performance_monitor import PerformanceAnalyzer

    print("✓ 成功导入所有增强模块")
except ImportError as e:
    print(f"⚠ 模块导入警告: {e}")
    print("将使用内置简化版本")

# 尝试导入TQSDK
try:
    from tqsdk import TqApi, TqAuth
    from tqsdk.ta import MA, MACD, RSI, BOLL

    TQSDK_AVAILABLE = True
    print("✓ TQSDK和技术指标库可用")
except ImportError:
    TQSDK_AVAILABLE = False
    print("⚠ TQSDK不可用，将使用内置技术指标计算")


class EnhancedAIStrategy:
    """增强版AI智能体策略系统"""

    def __init__(self, config_file: str = "enhanced_ai_strategy_config.json"):
        self.config_file = config_file
        self.config = self.load_config()

        # 初始化组件
        self.data_manager = None
        self.glm_client = None
        self.optimizer = None
        self.performance_analyzer = None

        # 策略状态
        self.positions = {}
        self.cash = 1000000.0  # 初始资金100万
        self.total_value = 1000000.0
        self.daily_pnl = 0.0

        # 初始化系统
        self.initialize_system()

        print("增强版AI智能体策略系统初始化完成")

    def load_config(self) -> Dict:
        """加载策略配置"""
        default_config = {
            "strategy": {
                "name": "增强版AI智能体策略",
                "version": "2.0.0",
                "description": "整合TQSDK、通达信数据源和GLM-4.7-flash的AI策略",
            },
            "trading": {
                "initial_cash": 1000000,
                "max_positions": 20,
                "position_size": 0.05,
                "rebalance_frequency": "daily",
                "trading_days": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                ],
            },
            "ai_analysis": {
                "model": "glm-4.7-flash",
                "min_ai_score": 1.5,
                "min_confidence": 0.6,
                "enable_real_ai": True,
                "fallback_to_simulation": True,
            },
            "risk_management": {
                "stop_loss": 0.10,
                "profit_target": 0.15,
                "max_drawdown": 0.20,
                "max_holding_days": 10,
                "force_exit_days": 10,
            },
            "data_sources": {
                "primary": "tqsdk",
                "backup": "tdx_local",
                "enable_caching": True,
                "cache_ttl": 300,
            },
            "optimization": {
                "enabled": True,
                "frequency": "weekly",
                "method": "grid_search",
            },
            "monitoring": {
                "enabled": True,
                "real_time_alerts": True,
                "performance_tracking": True,
            },
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                config = {**default_config, **loaded_config}
                print(f"✓ 已加载策略配置: {self.config_file}")
            else:
                config = default_config
                self.save_config(config)
                print(f"✓ 创建默认策略配置: {self.config_file}")
        except Exception as e:
            print(f"⚠ 策略配置加载失败，使用默认配置: {e}")
            config = default_config

        return config

    def save_config(self, config: Dict = None):
        """保存策略配置"""
        if config is None:
            config = self.config

        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✓ 策略配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"⚠ 策略配置保存失败: {e}")

    def initialize_system(self):
        """初始化系统组件"""
        print("\n正在初始化增强版AI策略系统...")

        # 初始化数据源管理器
        try:
            self.data_manager = DataSourceManager()
            print("✓ 数据源管理器初始化完成")
        except Exception as e:
            print(f"⚠ 数据源管理器初始化失败: {e}")

        # 初始化GLM客户端
        try:
            self.glm_client = GLMFlashClient()
            print("✓ GLM-4.7-flash客户端初始化完成")
        except Exception as e:
            print(f"⚠ GLM客户端初始化失败: {e}")

        # 初始化参数优化器
        try:
            if "ParameterOptimizer" in globals():
                self.optimizer = ParameterOptimizer()
                print("✓ 参数优化器初始化完成")
        except Exception as e:
            print(f"⚠ 参数优化器初始化失败: {e}")

        # 初始化性能监控器
        try:
            if "PerformanceAnalyzer" in globals():
                self.performance_analyzer = PerformanceAnalyzer()
                print("✓ 性能监控器初始化完成")
        except Exception as e:
            print(f"⚠ 性能监控器初始化失败: {e}")

        # 设置初始资金
        self.cash = self.config["trading"]["initial_cash"]
        self.total_value = self.cash

        print("✓ 系统组件初始化完成")

    def run_strategy(self, start_date: str = None, end_date: str = None) -> Dict:
        """运行策略"""
        print("\n" + "=" * 80)
        print("增强版AI智能体策略系统启动")
        print("=" * 80)

        start_time = time.time()

        # 设置日期范围
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"策略运行期间: {start_date} 到 {end_date}")

        # 获取股票池
        stock_pool = self.get_stock_pool()
        if not stock_pool:
            return {"error": "无法获取股票池"}

        print(f"股票池大小: {len(stock_pool)}")

        # 批量获取股票数据
        print("正在批量获取股票数据...")
        stock_data_dict = self.batch_get_stock_data(stock_pool)

        if not stock_data_dict:
            return {"error": "无法获取股票数据"}

        print(f"成功获取 {len(stock_data_dict)} 只股票数据")

        # 运行AI分析
        print("正在进行AI分析...")
        ai_results = self.run_ai_analysis(stock_data_dict)

        # 生成交易信号
        print("正在生成交易信号...")
        trading_signals = self.generate_trading_signals(ai_results)

        # 执行回测
        print("正在执行策略回测...")
        backtest_results = self.run_backtest(trading_signals, stock_data_dict)

        # 性能分析
        print("正在进行性能分析...")
        performance_analysis = self.analyze_performance(backtest_results)

        # 生成报告
        end_time = time.time()
        duration = end_time - start_time

        report = {
            "strategy_info": {
                "name": self.config["strategy"]["name"],
                "version": self.config["strategy"]["version"],
                "run_time": duration,
                "period": f"{start_date} to {end_date}",
            },
            "stock_pool": {
                "size": len(stock_pool),
                "successful_data": len(stock_data_dict),
            },
            "ai_analysis": {
                "total_analyzed": len(ai_results),
                "avg_score": np.mean([r["ai_score"] for r in ai_results.values()])
                if ai_results
                else 0,
                "avg_confidence": np.mean(
                    [r["confidence"] for r in ai_results.values()]
                )
                if ai_results
                else 0,
            },
            "trading_signals": {
                "buy_signals": len(
                    [s for s in trading_signals if s["action"] == "BUY"]
                ),
                "sell_signals": len(
                    [s for s in trading_signals if s["action"] == "SELL"]
                ),
                "hold_signals": len(
                    [s for s in trading_signals if s["action"] == "HOLD"]
                ),
            },
            "backtest_results": backtest_results,
            "performance_analysis": performance_analysis,
            "recommendations": self.generate_recommendations(performance_analysis),
        }

        # 保存报告
        self.save_strategy_report(report)

        # 显示结果
        self.display_results(report)

        return report

    def get_stock_pool(self) -> List[str]:
        """获取股票池"""
        if self.data_manager:
            return self.data_manager.get_stock_pool()
        else:
            # 使用默认股票池
            default_pool = [
                "000001.SZ",
                "000002.SZ",
                "000858.SZ",
                "002415.SZ",
                "002594.SZ",
                "600000.SH",
                "600036.SH",
                "600519.SH",
                "600887.SH",
                "601318.SH",
            ]
            print(f"使用默认股票池，共{len(default_pool)}只股票")
            return default_pool

    def batch_get_stock_data(self, stock_codes: List[str]) -> Dict[str, pd.DataFrame]:
        """批量获取股票数据"""
        if self.data_manager:
            return self.data_manager.batch_get_stock_data(stock_codes)
        else:
            print("⚠ 数据源管理器不可用，返回空数据")
            return {}

    def run_ai_analysis(
        self, stock_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """运行AI分析"""
        ai_results = {}

        for stock_code, stock_data in stock_data_dict.items():
            if stock_data is None or stock_data.empty:
                continue

            try:
                # 计算技术因子
                factors = self.calculate_technical_factors(stock_data)

                # 准备股票信息
                stock_info = {
                    "stock_code": stock_code,
                    "current_price": stock_data.iloc[-1]["close"],
                    "price_change_pct": self.calculate_price_change_pct(stock_data),
                }

                # AI分析
                if self.glm_client and self.config["ai_analysis"]["enable_real_ai"]:
                    ai_result = self.glm_client.analyze_stock(stock_info, factors)
                else:
                    ai_result = self.simulate_ai_analysis(stock_info, factors)

                ai_results[stock_code] = ai_result

            except Exception as e:
                print(f"⚠ {stock_code} AI分析失败: {e}")
                continue

        return ai_results

    def calculate_technical_factors(self, stock_data: pd.DataFrame) -> Dict:
        """计算技术因子"""
        if stock_data is None or len(stock_data) < 20:
            return {}

        close_prices = stock_data["close"].values
        volumes = stock_data["volume"].values
        high_prices = stock_data["high"].values
        low_prices = stock_data["low"].values

        current_price = close_prices[-1]

        factors = {}

        # 动量因子
        if len(close_prices) > 5:
            factors["momentum_5d"] = (current_price - close_prices[-6]) / close_prices[
                -6
            ]
        if len(close_prices) > 20:
            factors["momentum_20d"] = (
                current_price - close_prices[-21]
            ) / close_prices[-21]

        # RSI
        if len(close_prices) > 14:
            factors["rsi"] = self.calculate_rsi(close_prices)

        # 布尔带
        if len(close_prices) > 20:
            factors["bollinger_position"] = self.calculate_bollinger_position(
                close_prices
            )

        # 成交量因子
        if len(volumes) > 20:
            factors["volume_ratio"] = volumes[-1] / np.mean(volumes[-20:])
            factors["volume_trend"] = self.calculate_volume_trend(volumes)

        # 价格位置
        if len(high_prices) >= 20 and len(low_prices) >= 20:
            high_20d = np.max(high_prices[-20:])
            low_20d = np.min(low_prices[-20:])
            factors["price_position"] = (
                (current_price - low_20d) / (high_20d - low_20d)
                if high_20d != low_20d
                else 0.5
            )

        # 趋势因子
        if len(close_prices) >= 20:
            factors["trend_factor"] = self.calculate_trend_factor(close_prices)

        # 波动率因子
        if len(close_prices) > 20:
            returns = np.diff(close_prices[-20:]) / close_prices[-20:-1]
            factors["volatility"] = np.std(returns)

        return factors

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50.0

        delta = np.diff(prices[-period - 1 :])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain) if len(gain) > 0 else 0
        avg_loss = np.mean(loss) if len(loss) > 0 else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_position(
        self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0
    ) -> float:
        """计算布林带位置"""
        if len(prices) < period:
            return 0.5

        ma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        current_price = prices[-1]
        upper_band = ma + std_dev * std
        lower_band = ma - std_dev * std

        if upper_band == lower_band:
            return 0.5

        position = (current_price - lower_band) / (upper_band - lower_band)
        return np.clip(position, 0, 1)

    def calculate_volume_trend(self, volumes: np.ndarray, period: int = 10) -> float:
        """计算成交量趋势"""
        if len(volumes) < period:
            return 0.0

        recent_avg = np.mean(volumes[-period:])
        earlier_avg = (
            np.mean(volumes[-period * 2 : -period])
            if len(volumes) >= period * 2
            else recent_avg
        )

        if earlier_avg == 0:
            return 0.0

        trend = (recent_avg - earlier_avg) / earlier_avg
        return trend

    def calculate_trend_factor(
        self, prices: np.ndarray, short_period: int = 5, long_period: int = 20
    ) -> float:
        """计算趋势因子"""
        if len(prices) < long_period:
            return 0.0

        short_ma = np.mean(prices[-short_period:])
        long_ma = np.mean(prices[-long_period:])

        if long_ma == 0:
            return 0.0

        trend = (short_ma - long_ma) / long_ma
        return trend

    def calculate_price_change_pct(self, stock_data: pd.DataFrame) -> float:
        """计算价格变化百分比"""
        if len(stock_data) < 2:
            return 0.0

        current_price = stock_data.iloc[-1]["close"]
        previous_price = stock_data.iloc[-2]["close"]

        if previous_price == 0:
            return 0.0

        change_pct = (current_price - previous_price) / previous_price
        return change_pct

    def simulate_ai_analysis(self, stock_info: Dict, factors: Dict) -> Dict:
        """模拟AI分析（当真实AI不可用时）"""
        # 基于因子计算模拟评分
        score = 0.0
        score_components = []

        # 动量评分 (30%)
        momentum_5d = factors.get("momentum_5d", 0)
        momentum_20d = factors.get("momentum_20d", 0)
        momentum_score = (momentum_5d * 0.6 + momentum_20d * 0.4) * 100
        score += momentum_score * 0.3
        score_components.append(("momentum", momentum_score, 0.3))

        # RSI评分 (25%)
        rsi = factors.get("rsi", 50)
        rsi_score = (50 - rsi) * 0.5  # RSI超买超卖反向评分
        score += rsi_score * 0.25
        score_components.append(("rsi", rsi_score, 0.25))

        # 成交量评分 (20%)
        volume_ratio = factors.get("volume_ratio", 1)
        volume_score = (volume_ratio - 1) * 50
        score += volume_score * 0.2
        score_components.append(("volume", volume_score, 0.2))

        # 价格位置评分 (15%)
        price_position = factors.get("price_position", 0.5)
        position_score = (price_position - 0.5) * 10
        score += position_score * 0.15
        score_components.append(("position", position_score, 0.15))

        # 趋势评分 (10%)
        trend_factor = factors.get("trend_factor", 0)
        trend_score = trend_factor * 100
        score += trend_score * 0.1
        score_components.append(("trend", trend_score, 0.1))

        # 限制评分范围
        final_score = np.clip(score, -5, 5)

        # 计算置信度
        score_values = [comp[1] for comp in score_components]
        factor_consistency = 1 - (
            np.std(score_values) / (np.abs(np.mean(score_values)) + 0.1)
        )
        confidence = np.clip(factor_consistency * 0.8, 0.3, 1.0)

        # 生成推理说明
        reasoning_parts = [f"{comp[0]}:{comp[1]:.2f}" for comp in score_components]
        reasoning = (
            f"模拟AI分析: {', '.join(reasoning_parts)} | 置信度:{confidence:.2f}"
        )

        return {
            "ai_score": final_score,
            "confidence": confidence,
            "reasoning": reasoning,
            "recommendation": "BUY"
            if final_score > 1.5
            else "SELL"
            if final_score < -1.5
            else "HOLD",
            "risk_level": "HIGH" if abs(final_score) > 3 else "MEDIUM",
            "api_source": "simulation",
        }

    def generate_trading_signals(self, ai_results: Dict[str, Dict]) -> List[Dict]:
        """生成交易信号"""
        signals = []
        min_score = self.config["ai_analysis"]["min_ai_score"]
        min_confidence = self.config["ai_analysis"]["min_confidence"]

        for stock_code, ai_result in ai_results.items():
            ai_score = ai_result.get("ai_score", 0)
            confidence = ai_result.get("confidence", 0)

            # 生成信号
            if ai_score >= min_score and confidence >= min_confidence:
                action = "BUY"
                signal_strength = (
                    "STRONG" if ai_score > 3 else "MODERATE" if ai_score > 2 else "WEAK"
                )
            elif ai_score <= -min_score and confidence >= min_confidence:
                action = "SELL"
                signal_strength = (
                    "STRONG"
                    if ai_score < -3
                    else "MODERATE"
                    if ai_score < -2
                    else "WEAK"
                )
            else:
                action = "HOLD"
                signal_strength = "NEUTRAL"

            signals.append(
                {
                    "stock_code": stock_code,
                    "action": action,
                    "ai_score": ai_score,
                    "confidence": confidence,
                    "signal_strength": signal_strength,
                    "reasoning": ai_result.get("reasoning", ""),
                    "recommendation": ai_result.get("recommendation", "HOLD"),
                    "risk_level": ai_result.get("risk_level", "MEDIUM"),
                    "api_source": ai_result.get("api_source", "unknown"),
                }
            )

        # 按评分排序
        signals.sort(key=lambda x: x["ai_score"], reverse=True)

        return signals

    def run_backtest(
        self, trading_signals: List[Dict], stock_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict:
        """运行回测"""
        backtest_results = {
            "initial_cash": self.cash,
            "final_cash": self.cash,
            "total_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_holding_period": 0.0,
            "equity_curve": [],
            "trades": [],
        }

        # 模拟回测过程
        current_cash = self.cash
        positions = {}
        equity_curve = [current_cash]
        trades = []

        # 按日期排序信号（这里简化处理）
        daily_signals = {}
        for signal in trading_signals:
            if signal["action"] in ["BUY", "SELL"]:
                daily_signals[signal["stock_code"]] = signal

        # 模拟每日交易
        for day in range(60):  # 模拟60个交易日
            daily_pnl = 0.0

            # 检查持仓
            for stock_code, position_info in list(positions.items()):
                if stock_code in stock_data_dict:
                    stock_data = stock_data_dict[stock_code]
                    if len(stock_data) > day:
                        current_price = stock_data.iloc[day]["close"]

                        # 计算持仓盈亏
                        position_pnl = (
                            current_price - position_info["entry_price"]
                        ) * position_info["shares"]
                        daily_pnl += position_pnl

                        # 检查卖出条件
                        days_held = day - position_info["entry_day"]
                        return_pct = (
                            current_price - position_info["entry_price"]
                        ) / position_info["entry_price"]

                        should_sell = (
                            return_pct
                            >= self.config["risk_management"]["profit_target"]
                            or return_pct
                            <= -self.config["risk_management"]["stop_loss"]
                            or days_held
                            >= self.config["risk_management"]["max_holding_days"]
                        )

                        if should_sell:
                            # 卖出
                            sell_value = current_price * position_info["shares"]
                            current_cash += sell_value

                            # 记录交易
                            trade_return = return_pct
                            trades.append(
                                {
                                    "stock_code": stock_code,
                                    "action": "SELL",
                                    "entry_price": position_info["entry_price"],
                                    "exit_price": current_price,
                                    "shares": position_info["shares"],
                                    "return": trade_return,
                                    "holding_days": days_held,
                                    "day": day,
                                }
                            )

                            del positions[stock_code]

            # 检查买入信号
            buy_signals = [s for s in trading_signals if s["action"] == "BUY"]
            max_positions = self.config["trading"]["max_positions"]
            position_size = self.config["trading"]["position_size"]

            if len(positions) < max_positions and buy_signals:
                # 选择评分最高的买入信号
                available_signals = [
                    s for s in buy_signals if s["stock_code"] not in positions
                ]
                if available_signals:
                    best_signal = available_signals[0]
                    stock_code = best_signal["stock_code"]

                    if (
                        stock_code in stock_data_dict
                        and len(stock_data_dict[stock_code]) > day
                    ):
                        stock_data = stock_data_dict[stock_code]
                        current_price = stock_data.iloc[day]["close"]

                        # 计算可买入股数
                        max_investment = current_cash * position_size
                        shares = int(max_investment / current_price)

                        if shares > 0 and current_cash >= shares * current_price:
                            # 买入
                            cost = shares * current_price
                            current_cash -= cost

                            positions[stock_code] = {
                                "shares": shares,
                                "entry_price": current_price,
                                "entry_day": day,
                            }

                            # 记录交易
                            trades.append(
                                {
                                    "stock_code": stock_code,
                                    "action": "BUY",
                                    "entry_price": current_price,
                                    "shares": shares,
                                    "day": day,
                                }
                            )

            # 计算总资产
            total_positions_value = 0.0
            for stock_code, position_info in positions.items():
                if (
                    stock_code in stock_data_dict
                    and len(stock_data_dict[stock_code]) > day
                ):
                    current_price = stock_data_dict[stock_code].iloc[day]["close"]
                    total_positions_value += current_price * position_info["shares"]

            total_value = current_cash + total_positions_value
            equity_curve.append(total_value)

        # 计算回测结果
        if equity_curve:
            final_value = equity_curve[-1]
            total_return = (final_value - self.cash) / self.cash

            # 最大回撤
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            max_drawdown = abs(np.min(drawdown))

            # 夏普比率
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # 交易统计
            sell_trades = [t for t in trades if t["action"] == "SELL"]
            winning_trades = [t for t in sell_trades if t["return"] > 0]
            losing_trades = [t for t in sell_trades if t["return"] <= 0]

            win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0.0
            avg_holding_period = (
                np.mean([t["holding_days"] for t in sell_trades])
                if sell_trades
                else 0.0
            )

            backtest_results.update(
                {
                    "final_cash": current_cash + total_positions_value,
                    "total_return": total_return,
                    "annual_return": total_return * (252 / 60),  # 假设60天约3个月
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "win_rate": win_rate,
                    "total_trades": len(trades),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "avg_holding_period": avg_holding_period,
                    "equity_curve": equity_curve,
                    "trades": trades,
                }
            )

        return backtest_results

    def analyze_performance(self, backtest_results: Dict) -> Dict:
        """分析性能"""
        analysis = {
            "performance_grade": "C",
            "strengths": [],
            "weaknesses": [],
            "risk_assessment": "MEDIUM",
            "improvement_suggestions": [],
        }

        # 评分系统
        score = 0

        # 收益评分
        annual_return = backtest_results.get("annual_return", 0)
        if annual_return > 0.20:
            score += 30
            analysis["strengths"].append("年化收益率优秀")
        elif annual_return > 0.10:
            score += 20
            analysis["strengths"].append("年化收益率良好")
        elif annual_return > 0.05:
            score += 10
        else:
            analysis["weaknesses"].append("年化收益率偏低")

        # 夏普比率评分
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0)
        if sharpe_ratio > 2.0:
            score += 25
            analysis["strengths"].append("风险调整收益优秀")
        elif sharpe_ratio > 1.0:
            score += 15
            analysis["strengths"].append("风险调整收益良好")
        elif sharpe_ratio > 0.5:
            score += 5
        else:
            analysis["weaknesses"].append("风险调整收益偏低")

        # 最大回撤评分
        max_drawdown = backtest_results.get("max_drawdown", 0)
        if max_drawdown < 0.10:
            score += 20
            analysis["strengths"].append("回撤控制优秀")
        elif max_drawdown < 0.15:
            score += 10
            analysis["strengths"].append("回撤控制良好")
        else:
            analysis["weaknesses"].append("回撤控制需要改进")
            analysis["improvement_suggestions"].append("加强止损管理")

        # 胜率评分
        win_rate = backtest_results.get("win_rate", 0)
        if win_rate > 0.6:
            score += 15
            analysis["strengths"].append("胜率优秀")
        elif win_rate > 0.5:
            score += 10
            analysis["strengths"].append("胜率良好")
        elif win_rate > 0.4:
            score += 5
        else:
            analysis["weaknesses"].append("胜率偏低")
            analysis["improvement_suggestions"].append("提高信号质量")

        # 交易频率评分
        total_trades = backtest_results.get("total_trades", 0)
        if 20 <= total_trades <= 100:
            score += 10
            analysis["strengths"].append("交易频率适中")
        elif total_trades > 100:
            analysis["weaknesses"].append("交易过于频繁")
            analysis["improvement_suggestions"].append("降低交易频率")
        elif total_trades < 20:
            analysis["weaknesses"].append("交易频率过低")
            analysis["improvement_suggestions"].append("提高信号敏感度")

        # 评级
        if score >= 80:
            analysis["performance_grade"] = "A+"
        elif score >= 70:
            analysis["performance_grade"] = "A"
        elif score >= 60:
            analysis["performance_grade"] = "B"
        elif score >= 50:
            analysis["performance_grade"] = "C"
        else:
            analysis["performance_grade"] = "D"

        # 风险评估
        if max_drawdown > 0.20 or sharpe_ratio < 0.5:
            analysis["risk_assessment"] = "HIGH"
        elif max_drawdown > 0.15 or sharpe_ratio < 1.0:
            analysis["risk_assessment"] = "MEDIUM"
        else:
            analysis["risk_assessment"] = "LOW"

        return analysis

    def generate_recommendations(self, performance_analysis: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于性能分析生成建议
        if "年化收益率偏低" in performance_analysis.get("weaknesses", []):
            recommendations.append("考虑提高AI评分阈值，选择更高质量的交易信号")
            recommendations.append("优化仓位管理，适当增加高置信度信号的仓位")

        if "风险调整收益偏低" in performance_analysis.get("weaknesses", []):
            recommendations.append("加强风险控制，降低最大回撤")
            recommendations.append("优化因子权重，提升预测准确性")

        if "回撤控制需要改进" in performance_analysis.get("weaknesses", []):
            recommendations.append("设置更严格的止损条件")
            recommendations.append("降低最大持仓数量，减少组合风险")

        if "胜率偏低" in performance_analysis.get("weaknesses", []):
            recommendations.append("提高最小置信度要求")
            recommendations.append("优化AI模型参数，提升信号质量")

        if "交易过于频繁" in performance_analysis.get("weaknesses", []):
            recommendations.append("增加信号过滤条件，减少噪音交易")
            recommendations.append("延长最小持仓周期")

        if "交易频率过低" in performance_analysis.get("weaknesses", []):
            recommendations.append("降低AI评分阈值，增加交易机会")
            recommendations.append("优化因子敏感性，提高信号响应")

        # 通用建议
        recommendations.append("定期重新训练AI模型，适应市场变化")
        recommendations.append("监控策略表现，及时调整参数")
        recommendations.append("考虑市场环境因素，避免不利时期交易")

        return recommendations

    def save_strategy_report(self, report: Dict):
        """保存策略报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_ai_strategy_report_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✓ 策略报告已保存到: {filename}")
        except Exception as e:
            print(f"⚠ 策略报告保存失败: {e}")

    def display_results(self, report: Dict):
        """显示结果"""
        print("\n" + "=" * 80)
        print("策略运行结果")
        print("=" * 80)

        # 基本信息
        print(f"\n📊 策略信息:")
        print(f"  名称: {report['strategy_info']['name']}")
        print(f"  版本: {report['strategy_info']['version']}")
        print(f"  运行时间: {report['strategy_info']['run_time']:.2f}秒")
        print(f"  分析期间: {report['strategy_info']['period']}")

        # 股票池和数据
        print(f"\n📈 数据统计:")
        print(f"  股票池大小: {report['stock_pool']['size']}")
        print(f"  成功获取数据: {report['stock_pool']['successful_data']}")

        # AI分析
        print(f"\n🤖 AI分析:")
        print(f"  分析股票数: {report['ai_analysis']['total_analyzed']}")
        print(f"  平均AI评分: {report['ai_analysis']['avg_score']:.2f}")
        print(f"  平均置信度: {report['ai_analysis']['avg_confidence']:.2%}")

        # 交易信号
        print(f"\n📊 交易信号:")
        print(f"  买入信号: {report['trading_signals']['buy_signals']}")
        print(f"  卖出信号: {report['trading_signals']['sell_signals']}")
        print(f"  持有信号: {report['trading_signals']['hold_signals']}")

        # 回测结果
        backtest = report["backtest_results"]
        print(f"\n💰 回测结果:")
        print(f"  初始资金: ¥{backtest['initial_cash']:,.0f}")
        print(f"  最终资金: ¥{backtest['final_cash']:,.0f}")
        print(f"  总收益率: {backtest['total_return']:.2%}")
        print(f"  年化收益: {backtest['annual_return']:.2%}")
        print(f"  最大回撤: {backtest['max_drawdown']:.2%}")
        print(f"  夏普比率: {backtest['sharpe_ratio']:.2f}")
        print(f"  胜率: {backtest['win_rate']:.2%}")
        print(f"  总交易数: {backtest['total_trades']}")
        print(f"  平均持仓天数: {backtest['avg_holding_period']:.1f}")

        # 性能分析
        performance = report["performance_analysis"]
        print(f"\n🎯 性能分析:")
        print(f"  综合评级: {performance['performance_grade']}")
        print(f"  风险评估: {performance['risk_assessment']}")

        if performance.get("strengths"):
            print(f"  优势: {', '.join(performance['strengths'])}")

        if performance.get("weaknesses"):
            print(f"  不足: {', '.join(performance['weaknesses'])}")

        # 改进建议
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 80)

    def close(self):
        """关闭系统"""
        if self.data_manager:
            self.data_manager.close()
        print("✓ 增强版AI策略系统已关闭")


def main():
    """主函数"""
    print("增强版AI智能体策略系统")
    print("整合TQSDK、本地通达信数据源和GLM-4.7-flash模型")
    print("基于北京大学光华管理学院前沿研究")
    print("=" * 80)

    # 创建策略系统
    strategy = EnhancedAIStrategy()

    try:
        # 运行策略
        report = strategy.run_strategy()

        print("\n🎉 策略运行完成!")

    except KeyboardInterrupt:
        print("\n\n用户中断，正在关闭系统...")
    except Exception as e:
        print(f"\n⚠ 策略运行出错: {e}")
    finally:
        # 关闭系统
        strategy.close()

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
