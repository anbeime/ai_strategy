#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI智能体投资策略系统 - 快速测试脚本
简化版本，专注于核心功能验证
"""

import os
import sys
import json
import asyncio
import numpy as np
import pandas as pd
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleAIStrategy:
    """简化版AI策略"""

    def __init__(self):
        """初始化简化AI策略"""
        self.initial_cash = 100000.0
        self.max_positions = 10
        self.commission_rate = 0.0003

        # 状态
        self.cash = self.initial_cash
        self.positions = {}
        self.trade_history = []

        logger.info("简化AI策略初始化完成")

    def load_stock_pool(self, file_path: str = "C:/F/stock_pool.txt") -> List[str]:
        """加载股票池"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                stocks = [line.strip() for line in f if line.strip()]
            logger.info(f"加载股票池: {len(stocks)}只股票")
            return stocks[:20]  # 只取前20只进行测试
        except Exception as e:
            logger.error(f"加载股票池失败: {e}")
            return ["000001", "000002", "000003", "000004", "000005"]  # 默认股票

    def read_tdx_data(
        self, stock_code: str, date: datetime
    ) -> Optional[Dict[str, float]]:
        """读取通达信数据"""
        try:
            tdx_path = Path("C:/F/newtdx")
            vipdoc_path = tdx_path / "vipdoc"

            # 确定文件路径
            if stock_code.startswith("00") or stock_code.startswith("30"):
                file_path = vipdoc_path / "sz" / "lday" / f"sz{stock_code}.day"
            elif stock_code.startswith("6"):
                file_path = vipdoc_path / "sh" / "lday" / f"sh{stock_code}.day"
            else:
                return None

            if not file_path.exists():
                return None

            # 读取数据
            target_date_int = int(date.strftime("%Y%m%d"))

            with open(file_path, "rb") as f:
                while True:
                    buffer = f.read(32)
                    if len(buffer) < 32:
                        break

                    record = struct.unpack("<IIIIIfII", buffer)
                    date_int = record[0]

                    if date_int == target_date_int:
                        return {
                            "open": record[1] / 100.0,
                            "high": record[2] / 100.0,
                            "low": record[3] / 100.0,
                            "close": record[4] / 100.0,
                            "volume": record[5],
                        }

                    if date_int > target_date_int:
                        break

            return None

        except Exception as e:
            logger.error(f"读取{stock_code}数据失败: {e}")
            return None

    def generate_signal(
        self, stock_code: str, price_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """生成交易信号"""
        try:
            # 模拟AI分析
            np.random.seed(hash(stock_code) % 1000)  # 基于股票代码的随机种子

            # 基础评分（基于价格变化）
            price_change = (price_data["close"] - price_data["open"]) / price_data[
                "open"
            ]
            base_score = price_change * 10  # 放大到-5到5范围

            # 添加随机因素模拟AI分析
            ai_factor = np.random.normal(0, 1)
            final_score = np.clip(base_score + ai_factor, -5, 5)

            # 置信度
            confidence = max(0.3, 1.0 - abs(ai_factor) / 3)

            # 信号类型
            if final_score > 1.5:
                signal_type = "buy"
            elif final_score < -1.5:
                signal_type = "sell"
            else:
                signal_type = "hold"

            return {
                "symbol": stock_code,
                "signal_type": signal_type,
                "score": final_score,
                "confidence": confidence,
                "reasoning": f"价格变化: {price_change:.2%}, AI分析: {ai_factor:.2f}",
                "price": price_data["close"],
            }

        except Exception as e:
            logger.error(f"生成{stock_code}信号失败: {e}")
            return {
                "symbol": stock_code,
                "signal_type": "hold",
                "score": 0.0,
                "confidence": 0.0,
                "reasoning": f"信号生成失败: {str(e)}",
                "price": price_data.get("close", 10.0),
            }

    def make_trading_decision(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """做出交易决策"""
        # 策略：每日清仓，买入评分最高的股票

        # 1. 卖出所有持仓
        sell_orders = []
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                sell_orders.append(
                    {"symbol": symbol, "action": "sell", "quantity": quantity}
                )

        # 2. 选择买入信号
        buy_signals = [
            s
            for s in signals
            if s["signal_type"] == "buy" and s["score"] > 1.0 and s["confidence"] > 0.5
        ]
        buy_signals = sorted(buy_signals, key=lambda x: x["score"], reverse=True)[
            : self.max_positions
        ]

        # 3. 生成买入订单
        buy_orders = []
        if buy_signals and self.cash > 10000:  # 保留1万元缓冲
            equal_amount = (self.cash - 10000) / len(buy_signals)

            for signal in buy_signals:
                quantity = int(equal_amount / signal["price"] / 100) * 100  # 按手买入
                if quantity > 0:
                    buy_orders.append(
                        {
                            "symbol": signal["symbol"],
                            "action": "buy",
                            "quantity": quantity,
                            "price": signal["price"],
                            "score": signal["score"],
                            "reasoning": signal["reasoning"],
                        }
                    )

        return {
            "sell_orders": sell_orders,
            "buy_orders": buy_orders,
            "total_signals": len(signals),
            "buy_signals": len(buy_signals),
        }

    def execute_trades(self, orders: Dict[str, Any]) -> Dict[str, Any]:
        """执行交易"""
        execution_result = {
            "success": True,
            "executed_trades": [],
            "remaining_cash": self.cash,
        }

        # 执行卖出订单
        for order in orders["sell_orders"]:
            try:
                # 模拟卖出价格（随机波动）
                sell_price = 10.0 + np.random.normal(0, 0.5)
                proceeds = order["quantity"] * sell_price * (1 - self.commission_rate)

                self.cash += proceeds
                self.positions[order["symbol"]] = (
                    self.positions.get(order["symbol"], 0) - order["quantity"]
                )

                execution_result["executed_trades"].append(
                    {
                        "symbol": order["symbol"],
                        "action": "sell",
                        "quantity": order["quantity"],
                        "price": sell_price,
                        "proceeds": proceeds,
                    }
                )

            except Exception as e:
                logger.error(f"执行卖出{order['symbol']}失败: {e}")

        # 执行买入订单
        for order in orders["buy_orders"]:
            try:
                cost = order["quantity"] * order["price"] * (1 + self.commission_rate)

                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[order["symbol"]] = (
                        self.positions.get(order["symbol"], 0) + order["quantity"]
                    )

                    execution_result["executed_trades"].append(
                        {
                            "symbol": order["symbol"],
                            "action": "buy",
                            "quantity": order["quantity"],
                            "price": order["price"],
                            "cost": cost,
                            "score": order["score"],
                            "reasoning": order["reasoning"],
                        }
                    )
                else:
                    logger.warning(f"资金不足，无法买入{order['symbol']}")

            except Exception as e:
                logger.error(f"执行买入{order['symbol']}失败: {e}")

        execution_result["remaining_cash"] = self.cash
        return execution_result

    def calculate_total_assets(self, current_prices: Dict[str, float]) -> float:
        """计算总资产"""
        total_assets = self.cash

        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in current_prices:
                total_assets += quantity * current_prices[symbol]

        return total_assets

    async def run_backtest(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """运行回测"""
        logger.info(
            f"开始回测: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
        )

        # 加载股票池
        stock_codes = self.load_stock_pool()

        # 重置状态
        self.cash = self.initial_cash
        self.positions = {}
        self.trade_history = []

        # 生成交易日期
        trading_dates = []
        current_date = start_date

        while current_date <= end_date:
            if current_date.weekday() < 5:  # 周一到周五
                trading_dates.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"共{len(trading_dates)}个交易日")

        # 逐日回测
        daily_results = []

        for i, date in enumerate(trading_dates):
            if (i + 1) % 5 == 0:
                logger.info(
                    f"回测进度: {i + 1}/{len(trading_dates)} ({date.strftime('%Y-%m-%d')})"
                )

            # 1. 获取价格数据
            price_data = {}
            signals = []

            for symbol in stock_codes:
                data = self.read_tdx_data(symbol, date)
                if data:
                    price_data[symbol] = data["close"]
                    signal = self.generate_signal(symbol, data)
                    signals.append(signal)

            # 2. 做出交易决策
            decision = self.make_trading_decision(signals)

            # 3. 执行交易
            execution = self.execute_trades(decision)

            # 4. 计算当日资产
            total_assets = self.calculate_total_assets(price_data)
            daily_return = (total_assets / self.initial_cash - 1) * 100

            # 5. 记录结果
            daily_result = {
                "date": date.strftime("%Y-%m-%d"),
                "cash": self.cash,
                "positions": dict(self.positions),
                "total_assets": total_assets,
                "daily_return": daily_return,
                "trades": execution["executed_trades"],
                "signals_count": len(signals),
                "buy_signals": decision["buy_signals"],
            }

            daily_results.append(daily_result)

            # 记录交易历史
            for trade in execution["executed_trades"]:
                self.trade_history.append({"date": date.strftime("%Y-%m-%d"), **trade})

        # 计算最终统计
        final_stats = self._calculate_statistics(daily_results)

        logger.info(
            f"回测完成: 总收益率 {final_stats['total_return']:.2%}, 交易次数 {final_stats['total_trades']}"
        )

        return {
            "config": {
                "initial_cash": self.initial_cash,
                "max_positions": self.max_positions,
                "commission_rate": self.commission_rate,
            },
            "backtest_period": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "trading_days": len(trading_dates),
            },
            "final_statistics": final_stats,
            "daily_results": daily_results,
            "trade_history": self.trade_history,
        }

    def _calculate_statistics(
        self, daily_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算统计指标"""
        if not daily_results:
            return {}

        # 基本指标
        initial_assets = self.initial_cash
        final_assets = daily_results[-1]["total_assets"]
        total_return = final_assets / initial_assets - 1

        # 日收益率
        daily_returns = [r["daily_return"] for r in daily_results]

        # 年化收益率
        trading_days = len(daily_results)
        annualized_return = (
            (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        )

        # 波动率
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0

        # 夏普比率
        sharpe_ratio = (annualized_return - 0.03) / volatility if volatility > 0 else 0

        # 最大回撤
        max_drawdown = 0
        peak = initial_assets

        for result in daily_results:
            current_assets = result["total_assets"]
            if current_assets > peak:
                peak = current_assets
            else:
                drawdown = (peak - current_assets) / peak
                max_drawdown = max(max_drawdown, drawdown)

        # 交易统计
        total_trades = len(self.trade_history)
        buy_trades = len([t for t in self.trade_history if t["action"] == "buy"])
        sell_trades = len([t for t in self.trade_history if t["action"] == "sell"])

        return {
            "initial_cash": initial_assets,
            "final_assets": final_assets,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "volatility": volatility,
            "volatility_pct": volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "trading_days": trading_days,
        }


async def main():
    """主程序"""
    print("=" * 80)
    print("🤖 AI智能体投资策略系统 - 快速测试")
    print("基于北京大学光华管理学院前沿研究")
    print("=" * 80)

    # 初始化策略
    strategy = SimpleAIStrategy()

    # 设置回测期间
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    print(
        f"📅 回测期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"💰 初始资金: {strategy.initial_cash:,.2f} 元")
    print(f"🎯 最大持仓: {strategy.max_positions} 只")
    print(f"📊 手续费率: {strategy.commission_rate:.3%}")

    # 运行回测
    result = await strategy.run_backtest(start_date, end_date)

    # 输出结果
    print("\n" + "=" * 80)
    print("📊 回测结果")
    print("=" * 80)

    stats = result["final_statistics"]
    print(f"💰 初始资金: {stats['initial_cash']:,.2f} 元")
    print(f"💰 最终资产: {stats['final_assets']:,.2f} 元")
    print(f"📈 总收益率: {stats['total_return_pct']:.2f}%")
    print(f"📈 年化收益率: {stats['annualized_return_pct']:.2f}%")
    print(f"📊 夏普比率: {stats['sharpe_ratio']:.2f}")
    print(f"📉 最大回撤: {stats['max_drawdown_pct']:.2f}%")
    print(f"📊 波动率: {stats['volatility_pct']:.2f}%")
    print(f"📈 总交易次数: {stats['total_trades']}")
    print(f"📊 买入次数: {stats['buy_trades']}")
    print(f"📊 卖出次数: {stats['sell_trades']}")
    print(f"📅 交易天数: {stats['trading_days']}")

    # 显示最近几天的交易
    print(f"\n📋 最近5个交易日:")
    for result in result["daily_results"][-5:]:
        print(
            f"  {result['date']}: 资产 {result['total_assets']:.2f} 元, 收益 {result['daily_return']:.2f}%, 信号 {result['signals_count']} 个"
        )

    # 保存结果
    results_file = (
        f"ai_strategy_quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📄 详细结果已保存: {results_file}")

    # 策略评估
    print("\n" + "=" * 80)
    print("🎯 策略评估")
    print("=" * 80)

    if stats["sharpe_ratio"] > 1.0:
        print("✅ 策略表现优秀，具有实际投资价值")
    elif stats["sharpe_ratio"] > 0.5:
        print("⚠️ 策略表现中等，可考虑进一步优化")
    else:
        print("❌ 策略表现不佳，建议重新设计")

    if stats["max_drawdown"] < 0.1:
        print("✅ 风险控制良好")
    elif stats["max_drawdown"] < 0.2:
        print("⚠️ 风险控制中等")
    else:
        print("❌ 风险控制需要改进")

    print(f"\n🎉 快速测试完成！")
    print("\n💡 提示：")
    print("1. 这是简化版本的测试，实际应用中需要更复杂的数据和分析")
    print("2. 可以通过调整参数来优化策略表现")
    print("3. 建议在更长的时间周期上进行验证")
    print("4. 实盘交易前需要进行充分的风险评估")


if __name__ == "__main__":
    asyncio.run(main())
