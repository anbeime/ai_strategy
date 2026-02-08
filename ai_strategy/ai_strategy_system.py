#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI智能体策略执行与优化模块
完整的策略测试和验证系统
"""

import os
import sys
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """策略配置"""

    initial_cash: float = 100000.0
    max_positions: int = 20
    min_score: float = 1.0
    min_confidence: float = 0.6
    max_risk: float = 0.7
    commission_rate: float = 0.0003
    position_size_limit: float = 0.1
    tdx_path: str = "C:/F/newtdx"
    stock_pool_file: str = "C:/F/ai_strategy/stock_pool.txt"


@dataclass
class OptimizationResult:
    """优化结果"""

    best_config: StrategyConfig
    best_return: float
    best_sharpe: float
    best_max_drawdown: float
    optimization_history: List[Dict[str, Any]]
    backtest_results: Dict[str, Any]


class StrategyOptimizer:
    """策略优化器"""

    def __init__(self, base_config: StrategyConfig):
        """
        初始化策略优化器

        Args:
            base_config: 基础配置
        """
        self.base_config = base_config
        self.optimization_results = []

        logger.info("策略优化器初始化完成")

    async def optimize_strategy(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        param_grid: Dict[str, List[Any]] = None,
    ) -> OptimizationResult:
        """
        优化策略参数

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            param_grid: 参数网格

        Returns:
            优化结果
        """
        logger.info("开始策略参数优化")

        # 默认参数网格
        if param_grid is None:
            param_grid = {
                "max_positions": [10, 15, 20, 25],
                "min_score": [0.5, 1.0, 1.5, 2.0],
                "min_confidence": [0.5, 0.6, 0.7, 0.8],
                "max_risk": [0.6, 0.7, 0.8, 0.9],
            }

        # 生成参数组合
        param_combinations = self._generate_param_combinations(param_grid)

        logger.info(f"共生成 {len(param_combinations)} 个参数组合")

        best_result = None
        best_config = None
        optimization_history = []

        # 逐个测试参数组合
        for i, params in enumerate(param_combinations):
            logger.info(f"测试参数组合 {i + 1}/{len(param_combinations)}: {params}")

            # 创建配置
            config = self._create_config_from_params(params)

            # 运行回测
            try:
                from backtest_engine import BacktestEngine

                engine = BacktestEngine(asdict(config))
                result = await engine.run_backtest(stock_codes, start_date, end_date)

                # 记录结果
                history_item = {
                    "params": params,
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "annualized_return": result.annualized_return,
                }
                optimization_history.append(history_item)

                # 更新最佳结果
                if (
                    best_result is None
                    or result.sharpe_ratio > best_result.sharpe_ratio
                    or (
                        result.sharpe_ratio == best_result.sharpe_ratio
                        and result.total_return > best_result.total_return
                    )
                ):
                    best_result = result
                    best_config = config

                logger.info(
                    f"  收益率: {result.total_return:.2%}, 夏普比率: {result.sharpe_ratio:.2f}, 最大回撤: {result.max_drawdown:.2%}"
                )

            except Exception as e:
                logger.error(f"参数组合 {params} 测试失败: {e}")
                continue

        if best_result is None:
            logger.error("所有参数组合测试失败")
            return OptimizationResult(
                best_config=self.base_config,
                best_return=0.0,
                best_sharpe=0.0,
                best_max_drawdown=0.0,
                optimization_history=[],
                backtest_results={},
            )

        logger.info(
            f"优化完成: 最佳夏普比率 {best_result.sharpe_ratio:.2f}, 最佳收益率 {best_result.total_return:.2%}"
        )

        return OptimizationResult(
            best_config=best_config,
            best_return=best_result.total_return,
            best_sharpe=best_result.sharpe_ratio,
            best_max_drawdown=best_result.max_drawdown,
            optimization_history=optimization_history,
            backtest_results=asdict(best_result),
        )

    def _generate_param_combinations(
        self, param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """生成参数组合"""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def _create_config_from_params(self, params: Dict[str, Any]) -> StrategyConfig:
        """从参数创建配置"""
        config = StrategyConfig()

        # 更新配置
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def plot_optimization_results(
        self, result: OptimizationResult, save_path: str = None
    ):
        """
        绘制优化结果

        Args:
            result: 优化结果
            save_path: 保存路径
        """
        try:
            history = result.optimization_history

            if not history:
                logger.warning("没有优化历史数据")
                return

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("AI智能体策略参数优化结果", fontsize=16)

            # 1. 收益率分布
            returns = [item["total_return"] for item in history]
            axes[0, 0].hist(returns, bins=20, alpha=0.7, color="blue")
            axes[0, 0].axvline(
                result.best_return,
                color="red",
                linestyle="--",
                label=f"最佳: {result.best_return:.2%}",
            )
            axes[0, 0].set_title("总收益率分布")
            axes[0, 0].set_xlabel("收益率")
            axes[0, 0].set_ylabel("频次")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 夏普比率分布
            sharpe_ratios = [item["sharpe_ratio"] for item in history]
            axes[0, 1].hist(sharpe_ratios, bins=20, alpha=0.7, color="green")
            axes[0, 1].axvline(
                result.best_sharpe,
                color="red",
                linestyle="--",
                label=f"最佳: {result.best_sharpe:.2f}",
            )
            axes[0, 1].set_title("夏普比率分布")
            axes[0, 1].set_xlabel("夏普比率")
            axes[0, 1].set_ylabel("频次")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 收益率 vs 夏普比率散点图
            axes[1, 0].scatter(returns, sharpe_ratios, alpha=0.6, color="purple")
            axes[1, 0].scatter(
                result.best_return,
                result.best_sharpe,
                color="red",
                s=100,
                marker="*",
                label="最佳参数",
            )
            axes[1, 0].set_title("收益率 vs 夏普比率")
            axes[1, 0].set_xlabel("总收益率")
            axes[1, 0].set_ylabel("夏普比率")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 参数影响分析
            param_names = list(history[0]["params"].keys())
            param_returns = {}

            for param in param_names:
                param_values = [item["params"][param] for item in history]
                param_returns[param] = [item["total_return"] for item in history]

                # 计算每个参数值的平均收益率
                unique_values = list(set(param_values))
                avg_returns = []

                for value in unique_values:
                    indices = [i for i, v in enumerate(param_values) if v == value]
                    avg_return = np.mean([history[i]["total_return"] for i in indices])
                    avg_returns.append(avg_return)

                axes[1, 1].plot(unique_values, avg_returns, marker="o", label=param)

            axes[1, 1].set_title("参数对收益率的影响")
            axes[1, 1].set_xlabel("参数值")
            axes[1, 1].set_ylabel("平均收益率")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"优化结果图表已保存: {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"绘制优化结果失败: {e}")


class StrategyValidator:
    """策略验证器"""

    def __init__(self, config: StrategyConfig):
        """
        初始化策略验证器

        Args:
            config: 策略配置
        """
        self.config = config
        logger.info("策略验证器初始化完成")

    async def validate_strategy(
        self,
        stock_codes: List[str],
        validation_periods: List[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        验证策略

        Args:
            stock_codes: 股票代码列表
            validation_periods: 验证期间列表

        Returns:
            验证结果
        """
        logger.info("开始策略验证")

        # 默认验证期间
        if validation_periods is None:
            end_date = datetime.now()
            validation_periods = [
                (end_date - timedelta(days=90), end_date),  # 最近3个月
                (
                    end_date - timedelta(days=180),
                    end_date - timedelta(days=90),
                ),  # 3-6个月前
                (
                    end_date - timedelta(days=270),
                    end_date - timedelta(days=180),
                ),  # 6-9个月前
            ]

        validation_results = []

        for i, (start_date, end_date) in enumerate(validation_periods):
            logger.info(
                f"验证期间 {i + 1}: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
            )

            try:
                from backtest_engine import BacktestEngine

                engine = BacktestEngine(asdict(self.config))
                result = await engine.run_backtest(stock_codes, start_date, end_date)

                validation_results.append(
                    {
                        "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                        "total_return": result.total_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown,
                        "win_rate": result.win_rate,
                        "annualized_return": result.annualized_return,
                        "volatility": result.volatility,
                        "total_trades": result.total_trades,
                    }
                )

                logger.info(
                    f"  收益率: {result.total_return:.2%}, 夏普比率: {result.sharpe_ratio:.2f}"
                )

            except Exception as e:
                logger.error(f"验证期间 {i + 1} 失败: {e}")
                validation_results.append(
                    {
                        "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                        "error": str(e),
                    }
                )

        # 计算验证统计
        validation_summary = self._calculate_validation_summary(validation_results)

        logger.info(
            f"策略验证完成: 平均收益率 {validation_summary['avg_return']:.2%}, 夏普比率稳定性 {validation_summary['sharpe_stability']:.2f}"
        )

        return {
            "validation_results": validation_results,
            "validation_summary": validation_summary,
            "config": asdict(self.config),
        }

    def _calculate_validation_summary(
        self, validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算验证摘要"""
        successful_results = [r for r in validation_results if "error" not in r]

        if not successful_results:
            return {
                "avg_return": 0.0,
                "avg_sharpe": 0.0,
                "avg_max_drawdown": 0.0,
                "avg_win_rate": 0.0,
                "return_stability": 0.0,
                "sharpe_stability": 0.0,
                "success_rate": 0.0,
            }

        returns = [r["total_return"] for r in successful_results]
        sharpe_ratios = [r["sharpe_ratio"] for r in successful_results]
        max_drawdowns = [r["max_drawdown"] for r in successful_results]
        win_rates = [r["win_rate"] for r in successful_results]

        return {
            "avg_return": np.mean(returns),
            "avg_sharpe": np.mean(sharpe_ratios),
            "avg_max_drawdown": np.mean(max_drawdowns),
            "avg_win_rate": np.mean(win_rates),
            "return_stability": 1
            - (
                np.std(returns) / np.abs(np.mean(returns))
                if np.mean(returns) != 0
                else 0
            ),
            "sharpe_stability": 1
            - (
                np.std(sharpe_ratios) / np.abs(np.mean(sharpe_ratios))
                if np.mean(sharpe_ratios) != 0
                else 0
            ),
            "success_rate": len(successful_results) / len(validation_results),
        }

    def plot_validation_results(
        self, validation_results: Dict[str, Any], save_path: str = None
    ):
        """
        绘制验证结果

        Args:
            validation_results: 验证结果
            save_path: 保存路径
        """
        try:
            results = validation_results["validation_results"]
            successful_results = [r for r in results if "error" not in r]

            if not successful_results:
                logger.warning("没有成功的验证结果")
                return

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("AI智能体策略验证结果", fontsize=16)

            periods = [r["period"] for r in successful_results]
            returns = [r["total_return"] for r in successful_results]
            sharpe_ratios = [r["sharpe_ratio"] for r in successful_results]
            max_drawdowns = [r["max_drawdown"] for r in successful_results]
            win_rates = [r["win_rate"] for r in successful_results]

            # 1. 各期间收益率
            axes[0, 0].bar(range(len(periods)), returns, color="blue", alpha=0.7)
            axes[0, 0].set_title("各验证期间收益率")
            axes[0, 0].set_xlabel("验证期间")
            axes[0, 0].set_ylabel("收益率")
            axes[0, 0].set_xticks(range(len(periods)))
            axes[0, 0].set_xticklabels(
                [f"期间{i + 1}" for i in range(len(periods))], rotation=45
            )
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 各期间夏普比率
            axes[0, 1].bar(range(len(periods)), sharpe_ratios, color="green", alpha=0.7)
            axes[0, 1].set_title("各验证期间夏普比率")
            axes[0, 1].set_xlabel("验证期间")
            axes[0, 1].set_ylabel("夏普比率")
            axes[0, 1].set_xticks(range(len(periods)))
            axes[0, 1].set_xticklabels(
                [f"期间{i + 1}" for i in range(len(periods))], rotation=45
            )
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 收益率 vs 最大回撤
            axes[1, 0].scatter(returns, max_drawdowns, s=100, alpha=0.7, color="red")
            axes[1, 0].set_title("收益率 vs 最大回撤")
            axes[1, 0].set_xlabel("收益率")
            axes[1, 0].set_ylabel("最大回撤")
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 综合表现雷达图
            categories = ["收益率", "夏普比率", "胜率", "稳定性"]

            # 标准化指标
            avg_return = np.mean(returns)
            avg_sharpe = np.mean(sharpe_ratios)
            avg_win_rate = np.mean(win_rates)
            return_stability = 1 - (
                np.std(returns) / np.abs(avg_return) if avg_return != 0 else 0
            )

            values = [
                max(0, min(1, (avg_return + 0.2) / 0.4)),  # 假设收益率范围-20%到20%
                max(0, min(1, (avg_sharpe + 2) / 4)),  # 假设夏普比率范围-2到2
                avg_win_rate,
                return_stability,
            ]

            # 绘制雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]

            axes[1, 1] = plt.subplot(2, 2, 4, projection="polar")
            axes[1, 1].plot(angles, values, "o-", linewidth=2, color="purple")
            axes[1, 1].fill(angles, values, alpha=0.25, color="purple")
            axes[1, 1].set_xticks(angles[:-1])
            axes[1, 1].set_xticklabels(categories)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title("策略综合表现")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"验证结果图表已保存: {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"绘制验证结果失败: {e}")


class AIStrategySystem:
    """AI智能体策略系统"""

    def __init__(self, config: StrategyConfig = None):
        """
        初始化AI策略系统

        Args:
            config: 策略配置
        """
        self.config = config or StrategyConfig()

        # 加载股票池
        self.stock_codes = self._load_stock_pool()

        logger.info(f"AI策略系统初始化完成，股票池: {len(self.stock_codes)}只")

    def _load_stock_pool(self) -> List[str]:
        """加载股票池"""
        try:
            with open(self.config.stock_pool_file, "r", encoding="utf-8") as f:
                stocks = [line.strip() for line in f if line.strip()]
            logger.info(f"成功加载股票池: {len(stocks)}只股票")
            return stocks
        except Exception as e:
            logger.error(f"加载股票池失败: {e}")
            return []

    async def run_complete_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        optimize: bool = True,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        运行完整分析

        Args:
            start_date: 开始日期
            end_date: 结束日期
            optimize: 是否优化参数
            validate: 是否验证策略

        Returns:
            完整分析结果
        """
        logger.info("开始AI智能体策略完整分析")

        # 选择部分股票进行测试（避免计算量过大）
        test_stocks = (
            self.stock_codes[:50] if len(self.stock_codes) > 50 else self.stock_codes
        )

        analysis_results = {
            "config": asdict(self.config),
            "test_stocks": test_stocks,
            "analysis_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "backtest_result": None,
            "optimization_result": None,
            "validation_result": None,
        }

        # 1. 基础回测
        logger.info("步骤1: 基础回测")
        try:
            from backtest_engine import BacktestEngine

            engine = BacktestEngine(asdict(self.config))
            backtest_result = await engine.run_backtest(
                test_stocks, start_date, end_date
            )
            analysis_results["backtest_result"] = asdict(backtest_result)

            logger.info(
                f"基础回测完成: 收益率 {backtest_result.total_return:.2%}, 夏普比率 {backtest_result.sharpe_ratio:.2f}"
            )

        except Exception as e:
            logger.error(f"基础回测失败: {e}")
            analysis_results["backtest_result"] = {"error": str(e)}

        # 2. 参数优化
        if optimize:
            logger.info("步骤2: 参数优化")
            try:
                optimizer = StrategyOptimizer(self.config)
                optimization_result = await optimizer.optimize_strategy(
                    test_stocks[:20], start_date, end_date
                )  # 用更少的股票优化
                analysis_results["optimization_result"] = asdict(optimization_result)

                # 使用优化后的配置进行验证
                self.config = optimization_result.best_config

                logger.info(
                    f"参数优化完成: 最佳夏普比率 {optimization_result.best_sharpe:.2f}"
                )

            except Exception as e:
                logger.error(f"参数优化失败: {e}")
                analysis_results["optimization_result"] = {"error": str(e)}

        # 3. 策略验证
        if validate:
            logger.info("步骤3: 策略验证")
            try:
                validator = StrategyValidator(self.config)
                validation_result = await validator.validate_strategy(test_stocks[:30])
                analysis_results["validation_result"] = validation_result

                logger.info(
                    f"策略验证完成: 平均收益率 {validation_result['validation_summary']['avg_return']:.2%}"
                )

            except Exception as e:
                logger.error(f"策略验证失败: {e}")
                analysis_results["validation_result"] = {"error": str(e)}

        # 4. 生成报告
        self._generate_analysis_report(analysis_results)

        logger.info("AI智能体策略完整分析完成")

        return analysis_results

    def _generate_analysis_report(self, results: Dict[str, Any]):
        """生成分析报告"""
        try:
            report = []
            report.append("# AI智能体投资策略分析报告")
            report.append("")
            report.append(f"## 分析概览")
            report.append(f"- 分析期间: {results['analysis_period']}")
            report.append(f"- 测试股票: {len(results['test_stocks'])}只")
            report.append(f"- 初始资金: {results['config']['initial_cash']:,.2f}元")
            report.append(f"- 最大持仓: {results['config']['max_positions']}只")
            report.append("")

            # 基础回测结果
            if results["backtest_result"] and "error" not in results["backtest_result"]:
                br = results["backtest_result"]
                report.append("## 基础回测结果")
                report.append(f"- 总收益率: {br['total_return']:.2%}")
                report.append(f"- 年化收益率: {br['annualized_return']:.2%}")
                report.append(f"- 夏普比率: {br['sharpe_ratio']:.2f}")
                report.append(f"- 最大回撤: {br['max_drawdown']:.2%}")
                report.append(f"- 胜率: {br['win_rate']:.2%}")
                report.append(f"- 总交易次数: {br['total_trades']}")
                report.append("")

            # 优化结果
            if (
                results["optimization_result"]
                and "error" not in results["optimization_result"]
            ):
                opt = results["optimization_result"]
                report.append("## 参数优化结果")
                report.append(f"- 最佳收益率: {opt['best_return']:.2%}")
                report.append(f"- 最佳夏普比率: {opt['best_sharpe']:.2f}")
                report.append(f"- 最佳最大回撤: {opt['best_max_drawdown']:.2%}")
                report.append(f"- 测试参数组合数: {len(opt['optimization_history'])}")
                report.append("")

            # 验证结果
            if (
                results["validation_result"]
                and "error" not in results["validation_result"]
            ):
                val = results["validation_result"]
                summary = val["validation_summary"]
                report.append("## 策略验证结果")
                report.append(f"- 平均收益率: {summary['avg_return']:.2%}")
                report.append(f"- 平均夏普比率: {summary['avg_sharpe']:.2f}")
                report.append(f"- 收益率稳定性: {summary['return_stability']:.2f}")
                report.append(f"- 夏普比率稳定性: {summary['sharpe_stability']:.2f}")
                report.append(f"- 验证成功率: {summary['success_rate']:.2%}")
                report.append("")

            # 结论
            report.append("## 投资结论")
            if results["backtest_result"] and "error" not in results["backtest_result"]:
                if results["backtest_result"]["sharpe_ratio"] > 1.0:
                    report.append("✅ 策略表现优秀，具有实际投资价值")
                elif results["backtest_result"]["sharpe_ratio"] > 0.5:
                    report.append("⚠️ 策略表现中等，可考虑进一步优化")
                else:
                    report.append("❌ 策略表现不佳，建议重新设计")

            report.append("")
            report.append("---")
            report.append(
                f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # 保存报告
            report_file = f"ai_strategy_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("\n".join(report))

            logger.info(f"分析报告已保存: {report_file}")

        except Exception as e:
            logger.error(f"生成分析报告失败: {e}")


# 主程序
async def main():
    """主程序"""
    print("=" * 80)
    print("🤖 AI智能体投资策略系统")
    print("基于北京大学光华管理学院前沿研究")
    print("=" * 80)

    # 初始化系统
    config = StrategyConfig(
        initial_cash=100000.0, max_positions=20, min_score=1.0, min_confidence=0.6
    )

    system = AIStrategySystem(config)

    if not system.stock_codes:
        print("❌ 无法加载股票池，请检查股票池文件")
        return

    # 设置分析期间
    end_date = datetime(2024, 1, 31)
    start_date = datetime(2024, 1, 1)

    print(
        f"📅 分析期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"💰 初始资金: {config.initial_cash:,.2f} 元")
    print(f"📊 股票池: {len(system.stock_codes)} 只股票")
    print(f"🎯 最大持仓: {config.max_positions} 只")

    # 运行完整分析
    results = await system.run_complete_analysis(
        start_date=start_date, end_date=end_date, optimize=True, validate=True
    )

    # 输出摘要
    print("\n" + "=" * 80)
    print("📊 分析结果摘要")
    print("=" * 80)

    if results["backtest_result"] and "error" not in results["backtest_result"]:
        br = results["backtest_result"]
        print(f"📈 基础回测:")
        print(f"   总收益率: {br['total_return']:.2%}")
        print(f"   夏普比率: {br['sharpe_ratio']:.2f}")
        print(f"   最大回撤: {br['max_drawdown']:.2%}")

    if results["optimization_result"] and "error" not in results["optimization_result"]:
        opt = results["optimization_result"]
        print(f"🔧 参数优化:")
        print(f"   最佳收益率: {opt['best_return']:.2%}")
        print(f"   最佳夏普比率: {opt['best_sharpe']:.2f}")

    if results["validation_result"] and "error" not in results["validation_result"]:
        val = results["validation_result"]["validation_summary"]
        print(f"✅ 策略验证:")
        print(f"   平均收益率: {val['avg_return']:.2%}")
        print(f"   夏普稳定性: {val['sharpe_stability']:.2f}")

    # 保存完整结果
    results_file = (
        f"ai_strategy_complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📄 完整结果已保存: {results_file}")
    print("\n🎉 AI智能体策略分析完成！")


if __name__ == "__main__":
    asyncio.run(main())
