#!/usr/bin/python
"""
AI智能体策略参数优化器
基于回测结果自动优化策略参数，寻找最佳参数组合
"""

import numpy as np
import pandas as pd
import itertools
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Any


class ParameterOptimizer:
    """AI策略参数优化器"""

    def __init__(self):
        self.optimization_results = []
        self.best_params = None
        self.best_score = 0

        # 参数优化范围
        self.param_ranges = {
            "min_ai_score": [0.5, 1.0, 1.5, 2.0, 2.5],
            "min_confidence": [0.4, 0.5, 0.6, 0.7, 0.8],
            "max_positions": [10, 15, 20, 25, 30],
            "position_size": [0.03, 0.04, 0.05, 0.06, 0.07],
            "stop_loss": [0.05, 0.08, 0.10, 0.12, 0.15],
            "profit_target": [0.10, 0.12, 0.15, 0.18, 0.20],
            "momentum_weight": [0.2, 0.25, 0.3, 0.35, 0.4],
            "volume_weight": [0.15, 0.2, 0.25, 0.3, 0.35],
            "technical_weight": [0.25, 0.3, 0.35, 0.4, 0.45],
        }

        # 优化目标权重
        self.optimization_weights = {
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.25,
            "win_rate": 0.2,
            "profit_loss_ratio": 0.15,
            "annual_return": 0.1,
        }

    def generate_parameter_combinations(self, mode: str = "grid") -> List[Dict]:
        """生成参数组合"""
        if mode == "grid":
            # 网格搜索
            keys = list(self.param_ranges.keys())
            values = list(self.param_ranges.values())
            combinations = []

            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                # 验证参数合理性
                if self.validate_parameters(param_dict):
                    combinations.append(param_dict)

            return combinations

        elif mode == "random":
            # 随机搜索
            combinations = []
            for _ in range(100):  # 生成100个随机组合
                param_dict = {}
                for key, values in self.param_ranges.items():
                    param_dict[key] = np.random.choice(values)

                if self.validate_parameters(param_dict):
                    combinations.append(param_dict)

            return combinations

        elif mode == "bayesian":
            # 贝叶斯优化（简化版）
            return self._bayesian_optimization()

    def validate_parameters(self, params: Dict) -> bool:
        """验证参数合理性"""
        # 检查仓位大小和最大持仓数的匹配
        total_exposure = params["max_positions"] * params["position_size"]
        if total_exposure > 0.95:  # 总仓位不超过95%
            return False

        # 检查止损和止盈的合理性
        if params["stop_loss"] >= params["profit_target"]:
            return False

        # 检查权重总和
        weight_sum = (
            params["momentum_weight"]
            + params["volume_weight"]
            + params["technical_weight"]
        )
        if abs(weight_sum - 1.0) > 0.1:  # 权重总和应在1.0附近
            return False

        return True

    def calculate_optimization_score(self, backtest_results: Dict) -> float:
        """计算优化得分"""
        # 标准化各项指标
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0)
        max_drawdown = backtest_results.get("max_drawdown", 1.0)
        win_rate = backtest_results.get("win_rate", 0)
        profit_loss_ratio = backtest_results.get("profit_loss_ratio", 0)
        annual_return = backtest_results.get("annual_return", 0)

        # 计算各项得分（越高越好）
        sharpe_score = np.clip(sharpe_ratio / 2.0, 0, 1)  # 假设2.0为优秀
        drawdown_score = np.clip((1.0 - max_drawdown) / 0.8, 0, 1)  # 假设20%为优秀
        win_rate_score = np.clip(win_rate / 0.6, 0, 1)  # 假设60%为优秀
        pl_ratio_score = np.clip(profit_loss_ratio / 2.0, 0, 1)  # 假设2.0为优秀
        return_score = np.clip(annual_return / 0.3, 0, 1)  # 假设30%为优秀

        # 加权计算总分
        total_score = (
            sharpe_score * self.optimization_weights["sharpe_ratio"]
            + drawdown_score * self.optimization_weights["max_drawdown"]
            + win_rate_score * self.optimization_weights["win_rate"]
            + pl_ratio_score * self.optimization_weights["profit_loss_ratio"]
            + return_score * self.optimization_weights["annual_return"]
        )

        return total_score

    def simulate_backtest(self, params: Dict, test_data: Dict) -> Dict:
        """模拟回测（简化版）"""
        # 这里应该调用实际的回测引擎
        # 为了演示，我们使用模拟结果

        np.random.seed(hash(str(params)) % 2**32)

        # 基于参数生成模拟结果
        base_return = 0.15  # 基础年化收益
        base_volatility = 0.25  # 基础波动率

        # 参数影响
        ai_score_impact = (params["min_ai_score"] - 1.5) * 0.05
        confidence_impact = (params["min_confidence"] - 0.6) * 0.1
        position_impact = (params["position_size"] - 0.05) * 0.5

        # 计算收益指标
        annual_return = (
            base_return + ai_score_impact + confidence_impact + position_impact
        )
        annual_return += np.random.normal(0, 0.05)  # 添加随机性

        # 计算风险指标
        sharpe_ratio = annual_return / base_volatility
        max_drawdown = min(0.3, base_volatility * (1 + np.random.normal(0, 0.2)))

        # 计算交易指标
        win_rate = min(
            0.7, 0.5 + params["min_confidence"] * 0.2 + np.random.normal(0, 0.1)
        )
        profit_loss_ratio = 1.5 + params["profit_target"] / params["stop_loss"] * 0.3

        return {
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "total_trades": int(100 + np.random.normal(0, 20)),
            "avg_holding_days": 5 + np.random.normal(0, 2),
        }

    def run_optimization(self, mode: str = "grid", max_iterations: int = 50) -> Dict:
        """运行参数优化"""
        print("开始AI智能体策略参数优化...")
        print(f"优化模式: {mode}")
        print(f"最大迭代次数: {max_iterations}")

        # 生成参数组合
        param_combinations = self.generate_parameter_combinations(mode)
        print(f"生成 {len(param_combinations)} 个参数组合")

        # 限制迭代次数
        if len(param_combinations) > max_iterations:
            param_combinations = param_combinations[:max_iterations]

        # 运行优化
        start_time = time.time()

        for i, params in enumerate(param_combinations):
            print(f"\n优化进度: {i + 1}/{len(param_combinations)}")
            print(f"当前参数: {params}")

            # 模拟回测
            backtest_results = self.simulate_backtest(params, {})

            # 计算优化得分
            optimization_score = self.calculate_optimization_score(backtest_results)

            # 保存结果
            result = {
                "params": params,
                "backtest_results": backtest_results,
                "optimization_score": optimization_score,
                "timestamp": datetime.now().isoformat(),
            }

            self.optimization_results.append(result)

            # 更新最佳参数
            if optimization_score > self.best_score:
                self.best_score = optimization_score
                self.best_params = params
                print(f"发现更好参数! 得分: {optimization_score:.4f}")

            # 显示当前结果
            print(f"年化收益: {backtest_results['annual_return']:.2%}")
            print(f"夏普比率: {backtest_results['sharpe_ratio']:.2f}")
            print(f"最大回撤: {backtest_results['max_drawdown']:.2%}")
            print(f"胜率: {backtest_results['win_rate']:.2%}")
            print(f"盈亏比: {backtest_results['profit_loss_ratio']:.2f}")
            print(f"优化得分: {optimization_score:.4f}")

        end_time = time.time()
        print(f"\n优化完成! 耗时: {end_time - start_time:.2f}秒")

        return self.generate_optimization_report()

    def generate_optimization_report(self) -> Dict:
        """生成优化报告"""
        if not self.optimization_results:
            return {"error": "没有优化结果"}

        # 排序结果
        sorted_results = sorted(
            self.optimization_results,
            key=lambda x: x["optimization_score"],
            reverse=True,
        )

        # 统计分析
        scores = [r["optimization_score"] for r in self.optimization_results]

        report = {
            "summary": {
                "total_combinations": len(self.optimization_results),
                "best_score": self.best_score,
                "avg_score": np.mean(scores),
                "score_std": np.std(scores),
                "optimization_time": datetime.now().isoformat(),
            },
            "best_parameters": self.best_params,
            "top_5_results": sorted_results[:5],
            "parameter_analysis": self._analyze_parameter_importance(),
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _analyze_parameter_importance(self) -> Dict:
        """分析参数重要性"""
        if len(self.optimization_results) < 10:
            return {"error": "样本数量不足"}

        # 计算各参数与得分的相关性
        param_importance = {}

        for param_name in self.param_ranges.keys():
            param_values = [r["params"][param_name] for r in self.optimization_results]
            scores = [r["optimization_score"] for r in self.optimization_results]

            correlation = np.corrcoef(param_values, scores)[0, 1]
            param_importance[param_name] = {
                "correlation": correlation,
                "importance": abs(correlation),
            }

        # 按重要性排序
        sorted_importance = sorted(
            param_importance.items(), key=lambda x: x[1]["importance"], reverse=True
        )

        return {
            "ranking": sorted_importance,
            "most_important": sorted_importance[0][0] if sorted_importance else None,
            "least_important": sorted_importance[-1][0] if sorted_importance else None,
        }

    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if not self.best_params:
            return ["需要先运行参数优化"]

        # 基于最佳参数给出建议
        if self.best_params["min_ai_score"] > 2.0:
            recommendations.append("AI评分阈值较高，可能错过较多机会，考虑适当降低")

        if self.best_params["min_confidence"] > 0.7:
            recommendations.append("置信度要求较高，信号频率较低，考虑平衡置信度与机会")

        if self.best_params["max_positions"] > 25:
            recommendations.append("持仓数量较多，注意管理复杂度，考虑适当集中")

        if self.best_params["position_size"] < 0.04:
            recommendations.append("单笔仓位较小，可能限制收益，考虑适当提高")

        # 风险管理建议
        if self.best_params["stop_loss"] > 0.12:
            recommendations.append("止损幅度较大，建议控制在10%以内")

        if self.best_params["profit_target"] < 0.12:
            recommendations.append("止盈目标较小，建议适当提高盈亏比")

        return recommendations

    def _bayesian_optimization(self) -> List[Dict]:
        """贝叶斯优化（简化版实现）"""
        # 这里实现简化的贝叶斯优化
        # 实际应用中可以使用专门的贝叶斯优化库

        combinations = []

        # 基于先验知识生成初始参数
        for _ in range(20):
            param_dict = {
                "min_ai_score": np.random.normal(1.5, 0.5),
                "min_confidence": np.random.normal(0.6, 0.1),
                "max_positions": int(np.random.normal(20, 5)),
                "position_size": np.random.normal(0.05, 0.01),
                "stop_loss": np.random.normal(0.1, 0.02),
                "profit_target": np.random.normal(0.15, 0.03),
                "momentum_weight": np.random.normal(0.3, 0.05),
                "volume_weight": np.random.normal(0.2, 0.05),
                "technical_weight": np.random.normal(0.5, 0.05),
            }

            # 约束到合理范围
            for key, values in self.param_ranges.items():
                param_dict[key] = np.clip(param_dict[key], min(values), max(values))
                if key in ["max_positions"]:
                    param_dict[key] = int(param_dict[key])

            if self.validate_parameters(param_dict):
                combinations.append(param_dict)

        return combinations

    def save_results(self, filename: str = "ai_strategy_optimization_results.json"):
        """保存优化结果"""
        results = {
            "optimization_results": self.optimization_results,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "param_ranges": self.param_ranges,
            "optimization_weights": self.optimization_weights,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"优化结果已保存到: {filename}")

    def load_results(self, filename: str = "ai_strategy_optimization_results.json"):
        """加载之前的优化结果"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                results = json.load(f)

            self.optimization_results = results["optimization_results"]
            self.best_params = results["best_params"]
            self.best_score = results["best_score"]

            print(f"已加载 {len(self.optimization_results)} 个优化结果")
            return True
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return False


def main():
    """主函数"""
    optimizer = ParameterOptimizer()

    print("=" * 60)
    print("AI智能体策略参数优化器")
    print("=" * 60)

    # 选择优化模式
    print("\n请选择优化模式:")
    print("1. 网格搜索 (全面但较慢)")
    print("2. 随机搜索 (快速但可能错过最优)")
    print("3. 贝叶斯优化 (智能推荐)")

    choice = input("请输入选择 (1-3): ").strip()

    mode_map = {"1": "grid", "2": "random", "3": "bayesian"}
    mode = mode_map.get(choice, "grid")

    # 设置迭代次数
    max_iterations = int(input("请输入最大迭代次数 (默认50): ").strip() or "50")

    # 运行优化
    report = optimizer.run_optimization(mode=mode, max_iterations=max_iterations)

    # 显示结果
    print("\n" + "=" * 60)
    print("优化结果报告")
    print("=" * 60)

    print(f"\n最佳参数组合:")
    for key, value in report["best_parameters"].items():
        print(f"  {key}: {value}")

    print(f"\n最佳得分: {report['summary']['best_score']:.4f}")
    print(f"平均得分: {report['summary']['avg_score']:.4f}")

    print(f"\n参数重要性分析:")
    for param_name, analysis in report["parameter_analysis"]["ranking"][:5]:
        print(f"  {param_name}: 相关性 {analysis['correlation']:.3f}")

    print(f"\n优化建议:")
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"  {i}. {recommendation}")

    # 保存结果
    optimizer.save_results()

    print("\n优化完成! 请查看详细结果文件。")


if __name__ == "__main__":
    main()
