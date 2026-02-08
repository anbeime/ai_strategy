#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI智能体信号生成与决策系统
基于北京大学光华管理学院研究的核心策略
"""

import os
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

from information_processor import InformationCollector, InformationProcessor, MarketInfo

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """交易信号"""

    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    score: float  # -5 到 +5
    confidence: float  # 0 到 1
    reasoning: str
    timestamp: datetime
    expected_return: float  # 预期收益率
    risk_level: float  # 风险等级 0 到 1


@dataclass
class PortfolioDecision:
    """投资组合决策"""

    date: datetime
    buy_signals: List[TradingSignal]
    sell_signals: List[TradingSignal]
    target_positions: Dict[str, int]  # 目标持仓
    position_weights: Dict[str, float]  # 持仓权重
    cash_allocation: float  # 现金配置比例
    total_risk: float  # 组合总风险


class SignalGenerator:
    """信号生成器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化信号生成器

        Args:
            config: 配置信息
        """
        self.config = config or {}

        # 信号生成参数
        self.min_score = self.config.get("min_score", 1.0)  # 最低评分
        self.min_confidence = self.config.get("min_confidence", 0.6)  # 最低置信度
        self.max_positions = self.config.get("max_positions", 20)  # 最大持仓数

        # 信息收集器和处理器
        self.info_collector = InformationCollector(config)
        self.info_processor = InformationProcessor(config)

        logger.info("信号生成器初始化完成")

    async def generate_single_signal(
        self, symbol: str, date: datetime, price_data: Dict[str, float]
    ) -> TradingSignal:
        """
        为单只股票生成交易信号

        Args:
            symbol: 股票代码
            date: 信号日期
            price_data: 价格数据

        Returns:
            交易信号
        """
        try:
            # 收集市场信息
            market_info = await self.info_collector.collect_all_information(
                symbol, date, price_data
            )

            # 计算综合评分
            scores = self.info_processor.calculate_comprehensive_score(market_info)

            # 生成信号
            signal_type = self._determine_signal_type(scores["score"])

            # 计算预期收益和风险
            expected_return = self._calculate_expected_return(scores, price_data)
            risk_level = self._calculate_risk_level(scores, market_info)

            # 生成推理说明
            reasoning = self._generate_reasoning(scores, market_info)

            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                score=scores["score"],
                confidence=scores["confidence"],
                reasoning=reasoning,
                timestamp=date,
                expected_return=expected_return,
                risk_level=risk_level,
            )

        except Exception as e:
            logger.error(f"生成{symbol}信号失败: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type="hold",
                score=0.0,
                confidence=0.0,
                reasoning=f"信号生成失败: {str(e)}",
                timestamp=date,
                expected_return=0.0,
                risk_level=1.0,
            )

    def _determine_signal_type(self, score: float) -> str:
        """
        确定信号类型

        Args:
            score: 综合评分

        Returns:
            信号类型
        """
        if score > 2.0:
            return "buy"
        elif score < -2.0:
            return "sell"
        else:
            return "hold"

    def _calculate_expected_return(
        self, scores: Dict[str, float], price_data: Dict[str, float]
    ) -> float:
        """
        计算预期收益率

        Args:
            scores: 评分数据
            price_data: 价格数据

        Returns:
            预期收益率
        """
        # 基于评分计算预期收益
        base_return = scores["score"] * 0.02  # 每分对应2%收益

        # 技术指标调整
        technical_adjustment = scores["technical_signal"] * 0.01

        # 基本面调整
        fundamental_adjustment = scores["fundamental_score"] * 0.015

        # 新闻情绪调整
        news_adjustment = scores["news_sentiment"] * 0.005

        expected_return = (
            base_return
            + technical_adjustment
            + fundamental_adjustment
            + news_adjustment
        )

        # 限制在合理范围内
        return max(-0.1, min(0.1, expected_return))

    def _calculate_risk_level(
        self, scores: Dict[str, float], market_info: MarketInfo
    ) -> float:
        """
        计算风险等级

        Args:
            scores: 评分数据
            market_info: 市场信息

        Returns:
            风险等级 (0 到 1)
        """
        risk_factors = []

        # 评分一致性风险（评分越不一致，风险越高）
        score_variance = np.var(
            [
                scores["news_sentiment"],
                scores["technical_signal"],
                scores["fundamental_score"],
            ]
        )
        consistency_risk = min(1.0, score_variance * 2)
        risk_factors.append(consistency_risk)

        # 置信度风险（置信度越低，风险越高）
        confidence_risk = 1.0 - scores["confidence"]
        risk_factors.append(confidence_risk)

        # 技术指标风险
        technical_risk = 0.0
        if "rsi" in market_info.technical_indicators:
            rsi = market_info.technical_indicators["rsi"]
            if rsi > 80 or rsi < 20:
                technical_risk = 0.3
            elif rsi > 70 or rsi < 30:
                technical_risk = 0.15
        risk_factors.append(technical_risk)

        # 波动率风险
        volatility_risk = 0.0
        if "volatility" in market_info.technical_indicators:
            volatility = market_info.technical_indicators["volatility"]
            volatility_risk = min(1.0, volatility * 20)  # 假设5%波动率为正常水平
        risk_factors.append(volatility_risk)

        # 综合风险等级
        total_risk = np.mean(risk_factors)
        return max(0.0, min(1.0, total_risk))

    def _generate_reasoning(
        self, scores: Dict[str, float], market_info: MarketInfo
    ) -> str:
        """
        生成推理说明

        Args:
            scores: 评分数据
            market_info: 市场信息

        Returns:
            推理说明
        """
        reasoning_parts = []

        # 综合评分说明
        reasoning_parts.append(f"综合评分: {scores['score']:.2f}")

        # 新闻情绪说明
        news_desc = (
            "正面"
            if scores["news_sentiment"] > 0.1
            else "负面"
            if scores["news_sentiment"] < -0.1
            else "中性"
        )
        reasoning_parts.append(
            f"新闻情绪: {news_desc} ({scores['news_sentiment']:.2f})"
        )

        # 技术信号说明
        tech_desc = (
            "看涨"
            if scores["technical_signal"] > 0.1
            else "看跌"
            if scores["technical_signal"] < -0.1
            else "中性"
        )
        reasoning_parts.append(
            f"技术信号: {tech_desc} ({scores['technical_signal']:.2f})"
        )

        # 基本面说明
        fund_desc = (
            "良好"
            if scores["fundamental_score"] > 0.1
            else "较差"
            if scores["fundamental_score"] < -0.1
            else "一般"
        )
        reasoning_parts.append(
            f"基本面: {fund_desc} ({scores['fundamental_score']:.2f})"
        )

        # 置信度说明
        reasoning_parts.append(f"置信度: {scores['confidence']:.1%}")

        # 市场上下文
        if market_info.market_context:
            reasoning_parts.append(f"市场环境: {market_info.market_context}")

        return "; ".join(reasoning_parts)

    async def generate_batch_signals(
        self,
        symbols: List[str],
        date: datetime,
        price_data_dict: Dict[str, Dict[str, float]],
    ) -> List[TradingSignal]:
        """
        批量生成交易信号

        Args:
            symbols: 股票代码列表
            date: 信号日期
            price_data_dict: 价格数据字典

        Returns:
            交易信号列表
        """
        logger.info(
            f"开始为{len(symbols)}只股票生成{date.strftime('%Y-%m-%d')}的交易信号"
        )

        signals = []

        # 并发生成信号（限制并发数）
        semaphore = asyncio.Semaphore(10)

        async def generate_with_semaphore(symbol):
            async with semaphore:
                price_data = price_data_dict.get(symbol, {})
                if not price_data:
                    # 使用默认价格数据
                    price_data = {
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.5,
                        "close": 10.2,
                        "volume": 1000000,
                    }

                return await self.generate_single_signal(symbol, date, price_data)

        tasks = [generate_with_semaphore(symbol) for symbol in symbols]
        signals = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤异常结果
        valid_signals = []
        for signal in signals:
            if isinstance(signal, TradingSignal):
                valid_signals.append(signal)
            else:
                logger.error(f"信号生成异常: {signal}")

        logger.info(f"成功生成{len(valid_signals)}个交易信号")
        return valid_signals


class DecisionEngine:
    """决策引擎"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化决策引擎

        Args:
            config: 配置信息
        """
        self.config = config or {}

        # 决策参数
        self.max_positions = self.config.get("max_positions", 20)
        self.min_score = self.config.get("min_score", 1.0)
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.max_risk = self.config.get("max_risk", 0.7)  # 最大风险等级
        self.position_size_limit = self.config.get(
            "position_size_limit", 0.1
        )  # 单个持仓上限

        logger.info("决策引擎初始化完成")

    def make_portfolio_decision(
        self,
        signals: List[TradingSignal],
        current_positions: Dict[str, int],
        cash: float,
        date: datetime,
    ) -> PortfolioDecision:
        """
        做出投资组合决策

        Args:
            signals: 交易信号列表
            current_positions: 当前持仓
            cash: 现金余额
            date: 决策日期

        Returns:
            投资组合决策
        """
        logger.info(f"开始为{date.strftime('%Y-%m-%d')}做出投资组合决策")

        # 1. 过滤有效信号
        valid_signals = self._filter_valid_signals(signals)

        # 2. 分离买入和卖出信号
        buy_signals = [s for s in valid_signals if s.signal_type == "buy"]
        sell_signals = [s for s in valid_signals if s.signal_type == "sell"]

        # 3. 排序买入信号（按评分和置信度）
        buy_signals = self._rank_buy_signals(buy_signals)

        # 4. 选择目标持仓
        target_positions = self._select_target_positions(
            buy_signals, current_positions, cash
        )

        # 5. 计算持仓权重
        position_weights = self._calculate_position_weights(target_positions, cash)

        # 6. 计算现金配置
        cash_allocation = self._calculate_cash_allocation(target_positions, cash)

        # 7. 计算组合总风险
        total_risk = self._calculate_portfolio_risk(target_positions, valid_signals)

        decision = PortfolioDecision(
            date=date,
            buy_signals=buy_signals[: self.max_positions],
            sell_signals=sell_signals,
            target_positions=target_positions,
            position_weights=position_weights,
            cash_allocation=cash_allocation,
            total_risk=total_risk,
        )

        logger.info(
            f"决策完成: 买入{len(decision.buy_signals)}只, 卖出{len(decision.sell_signals)}只, 目标持仓{len(target_positions)}只"
        )

        return decision

    def _filter_valid_signals(
        self, signals: List[TradingSignal]
    ) -> List[TradingSignal]:
        """过滤有效信号"""
        valid_signals = []

        for signal in signals:
            if (
                signal.score >= self.min_score
                and signal.confidence >= self.min_confidence
                and signal.risk_level <= self.max_risk
            ):
                valid_signals.append(signal)

        return valid_signals

    def _rank_buy_signals(
        self, buy_signals: List[TradingSignal]
    ) -> List[TradingSignal]:
        """排序买入信号"""
        # 综合评分 = 信号评分 * 置信度 - 风险惩罚
        for signal in buy_signals:
            signal.comprehensive_score = (
                signal.score * signal.confidence - signal.risk_level * 2
            )

        # 按综合评分降序排列
        ranked_signals = sorted(
            buy_signals, key=lambda x: x.comprehensive_score, reverse=True
        )

        return ranked_signals

    def _select_target_positions(
        self,
        buy_signals: List[TradingSignal],
        current_positions: Dict[str, int],
        cash: float,
    ) -> Dict[str, int]:
        """选择目标持仓"""
        target_positions = {}

        # 策略：每日清仓，只买入评分最高的股票
        available_cash = cash

        # 选择前N只股票
        top_signals = buy_signals[: self.max_positions]

        if not top_signals:
            return target_positions

        # 等权重分配资金
        equal_weight = available_cash / len(top_signals)

        for signal in top_signals:
            if available_cash <= 1000:  # 保留1000元缓冲
                break

            # 计算可买入数量（假设价格为10元）
            estimated_price = 10.0  # 实际应该获取真实价格
            max_shares = int(equal_weight / estimated_price / 100) * 100  # 按手买入

            if max_shares > 0:
                target_positions[signal.symbol] = max_shares
                available_cash -= max_shares * estimated_price

        return target_positions

    def _calculate_position_weights(
        self, target_positions: Dict[str, int], cash: float
    ) -> Dict[str, float]:
        """计算持仓权重"""
        if not target_positions:
            return {}

        total_value = cash + sum(
            shares * 10.0 for shares in target_positions.values()
        )  # 假设价格为10元
        weights = {}

        for symbol, shares in target_positions.items():
            position_value = shares * 10.0  # 假设价格为10元
            weights[symbol] = position_value / total_value

        return weights

    def _calculate_cash_allocation(
        self, target_positions: Dict[str, int], cash: float
    ) -> float:
        """计算现金配置比例"""
        if not target_positions:
            return 1.0

        total_position_value = sum(
            shares * 10.0 for shares in target_positions.values()
        )
        total_value = cash + total_position_value

        return cash / total_value

    def _calculate_portfolio_risk(
        self, target_positions: Dict[str, int], signals: List[TradingSignal]
    ) -> float:
        """计算组合总风险"""
        if not target_positions:
            return 0.0

        # 获取持仓股票的风险等级
        position_risks = []
        for symbol in target_positions.keys():
            symbol_signals = [s for s in signals if s.symbol == symbol]
            if symbol_signals:
                # 使用最高置信度信号的风险等级
                best_signal = max(symbol_signals, key=lambda x: x.confidence)
                position_risks.append(best_signal.risk_level)

        if position_risks:
            # 组合风险 = 平均风险 + 集中度风险
            avg_risk = np.mean(position_risks)
            concentration_risk = (
                len(target_positions) / self.max_positions * 0.1
            )  # 持仓越集中，风险越高
            total_risk = avg_risk + concentration_risk
            return min(1.0, total_risk)

        return 0.0


class TradingExecutor:
    """交易执行器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化交易执行器

        Args:
            config: 配置信息
        """
        self.config = config or {}
        self.commission_rate = self.config.get("commission_rate", 0.0003)  # 手续费率

        logger.info("交易执行器初始化完成")

    def execute_decision(
        self,
        decision: PortfolioDecision,
        current_positions: Dict[str, int],
        cash: float,
    ) -> Dict[str, Any]:
        """
        执行交易决策

        Args:
            decision: 投资组合决策
            current_positions: 当前持仓
            cash: 现金余额

        Returns:
            执行结果
        """
        logger.info(f"开始执行{decision.date.strftime('%Y-%m-%d')}的交易决策")

        execution_results = {
            "date": decision.date.strftime("%Y-%m-%d"),
            "buy_orders": [],
            "sell_orders": [],
            "executed_positions": {},
            "remaining_cash": cash,
            "total_cost": 0.0,
            "success": True,
        }

        try:
            # 1. 执行卖出订单（清仓）
            for symbol, quantity in current_positions.items():
                if quantity > 0:
                    sell_result = self._execute_sell(symbol, quantity)
                    execution_results["sell_orders"].append(sell_result)
                    execution_results["remaining_cash"] += sell_result["proceeds"]

            # 2. 执行买入订单
            for symbol, quantity in decision.target_positions.items():
                if quantity > 0:
                    buy_result = self._execute_buy(
                        symbol, quantity, execution_results["remaining_cash"]
                    )
                    if buy_result["success"]:
                        execution_results["buy_orders"].append(buy_result)
                        execution_results["remaining_cash"] -= buy_result["cost"]
                        execution_results["executed_positions"][symbol] = quantity
                    else:
                        logger.warning(f"买入{symbol}失败: {buy_result['reason']}")

            # 3. 计算总成本
            execution_results["total_cost"] = sum(
                order["cost"] for order in execution_results["buy_orders"]
            )

            logger.info(
                f"交易执行完成: 买入{len(execution_results['buy_orders'])}只, 卖出{len(execution_results['sell_orders'])}只"
            )

            return execution_results

        except Exception as e:
            logger.error(f"交易执行失败: {e}")
            execution_results["success"] = False
            execution_results["error"] = str(e)
            return execution_results

    def _execute_sell(self, symbol: str, quantity: int) -> Dict[str, Any]:
        """执行卖出"""
        try:
            # 模拟卖出价格
            sell_price = 10.0 + np.random.normal(0, 0.5)  # 假设价格在10元左右波动
            proceeds = quantity * sell_price * (1 - self.commission_rate)

            return {
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "price": sell_price,
                "proceeds": proceeds,
                "commission": quantity * sell_price * self.commission_rate,
                "success": True,
            }

        except Exception as e:
            logger.error(f"执行卖出{symbol}失败: {e}")
            return {
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "success": False,
                "reason": str(e),
            }

    def _execute_buy(
        self, symbol: str, quantity: int, available_cash: float
    ) -> Dict[str, Any]:
        """执行买入"""
        try:
            # 模拟买入价格
            buy_price = 10.0 + np.random.normal(0, 0.5)  # 假设价格在10元左右波动
            total_cost = quantity * buy_price * (1 + self.commission_rate)

            if total_cost > available_cash:
                return {
                    "symbol": symbol,
                    "action": "buy",
                    "quantity": quantity,
                    "success": False,
                    "reason": f"资金不足: 需要{total_cost:.2f}, 可用{available_cash:.2f}",
                }

            return {
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "price": buy_price,
                "cost": total_cost,
                "commission": quantity * buy_price * self.commission_rate,
                "success": True,
            }

        except Exception as e:
            logger.error(f"执行买入{symbol}失败: {e}")
            return {
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "success": False,
                "reason": str(e),
            }


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_signal_system():
        """测试信号系统"""
        # 配置
        config = {"max_positions": 5, "min_score": 0.5, "min_confidence": 0.5}

        # 初始化组件
        signal_generator = SignalGenerator(config)
        decision_engine = DecisionEngine(config)
        trading_executor = TradingExecutor(config)

        # 测试数据
        symbols = ["000001", "000002", "000003", "000004", "000005"]
        date = datetime(2024, 1, 15)

        # 模拟价格数据
        price_data_dict = {}
        for symbol in symbols:
            price_data_dict[symbol] = {
                "open": 10.0 + np.random.normal(0, 1),
                "high": 10.5 + np.random.normal(0, 1),
                "low": 9.5 + np.random.normal(0, 1),
                "close": 10.2 + np.random.normal(0, 1),
                "volume": 1000000 + int(np.random.normal(0, 100000)),
            }

        # 生成信号
        signals = await signal_generator.generate_batch_signals(
            symbols, date, price_data_dict
        )

        print(f"生成{len(signals)}个信号:")
        for signal in signals:
            print(
                f"  {signal.symbol}: {signal.signal_type}, 评分: {signal.score:.2f}, 置信度: {signal.confidence:.2f}"
            )

        # 做出决策
        current_positions = {"000001": 100, "000002": 200}
        cash = 50000.0

        decision = decision_engine.make_portfolio_decision(
            signals, current_positions, cash, date
        )

        print(f"\n投资组合决策:")
        print(f"  买入信号: {len(decision.buy_signals)}个")
        print(f"  卖出信号: {len(decision.sell_signals)}个")
        print(f"  目标持仓: {decision.target_positions}")
        print(f"  现金配置: {decision.cash_allocation:.2%}")
        print(f"  组合风险: {decision.total_risk:.2f}")

        # 执行交易
        execution_result = trading_executor.execute_decision(
            decision, current_positions, cash
        )

        print(f"\n交易执行结果:")
        print(f"  执行成功: {execution_result['success']}")
        print(f"  买入订单: {len(execution_result['buy_orders'])}个")
        print(f"  卖出订单: {len(execution_result['sell_orders'])}个")
        print(f"  剩余现金: {execution_result['remaining_cash']:.2f}")
        print(f"  执行持仓: {execution_result['executed_positions']}")

    # 运行测试
    asyncio.run(test_signal_system())
