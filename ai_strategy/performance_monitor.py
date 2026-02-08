#!/usr/bin/python
"""
AI智能体策略性能监控与分析系统
实时监控策略表现，提供详细的性能分析和优化建议
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""

    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0

    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 交易指标
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    avg_trade_return: float = 0.0
    total_trades: int = 0

    # 持仓指标
    avg_holding_period: float = 0.0
    max_positions: int = 0
    position_turnover: float = 0.0

    # AI指标
    avg_ai_score: float = 0.0
    ai_score_distribution: Dict[str, int] = None
    confidence_distribution: Dict[str, int] = None

    # 时间指标
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0

    def __post_init__(self):
        if self.ai_score_distribution is None:
            self.ai_score_distribution = {}
        if self.confidence_distribution is None:
            self.confidence_distribution = {}


class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self):
        self.metrics_history = []
        self.current_metrics = PerformanceMetrics()
        self.benchmark_metrics = PerformanceMetrics()

        # 分析配置
        self.analysis_config = {
            "performance_thresholds": {
                "excellent_sharpe": 2.0,
                "good_sharpe": 1.0,
                "excellent_win_rate": 0.6,
                "good_win_rate": 0.5,
                "max_acceptable_drawdown": 0.2,
                "min_acceptable_return": 0.1,
            },
            "risk_limits": {
                "max_drawdown": 0.15,
                "max_position_loss": 0.08,
                "max_daily_loss": 0.05,
            },
        }

    def calculate_performance_metrics(
        self,
        equity_curve: List[float],
        trades: List[Dict],
        positions: List[Dict],
        ai_signals: List[Dict] = None,
    ) -> PerformanceMetrics:
        """计算性能指标"""
        if not equity_curve:
            return PerformanceMetrics()

        metrics = PerformanceMetrics()

        # 基础收益指标
        initial_value = equity_curve[0]
        final_value = equity_curve[-1]
        metrics.total_return = (final_value - initial_value) / initial_value

        # 年化收益（假设252个交易日）
        trading_days = len(equity_curve)
        metrics.trading_days = trading_days
        if trading_days > 0:
            metrics.annual_return = metrics.total_return * (252 / trading_days)
            metrics.monthly_return = metrics.total_return * (21 / trading_days)

        # 风险指标
        returns = np.diff(equity_curve) / equity_curve[:-1]
        metrics.volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        metrics.max_drawdown = abs(np.min(drawdown))

        # 夏普比率
        if metrics.volatility > 0:
            metrics.sharpe_ratio = metrics.annual_return / metrics.volatility

        # 卡玛比率
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown

        # 交易指标
        if trades:
            trade_returns = [trade.get("return", 0) for trade in trades]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            metrics.total_trades = len(trades)
            metrics.win_rate = len(winning_trades) / len(trades) if trades else 0

            if winning_trades and losing_trades:
                avg_win = np.mean(winning_trades)
                avg_loss = abs(np.mean(losing_trades))
                metrics.profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

            metrics.avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        # 持仓指标
        if positions:
            holding_periods = [pos.get("holding_days", 0) for pos in positions]
            metrics.avg_holding_period = (
                np.mean(holding_periods) if holding_periods else 0
            )

            position_counts = [pos.get("position_count", 0) for pos in positions]
            metrics.max_positions = max(position_counts) if position_counts else 0

            # 换手率
            if metrics.total_trades > 0 and trading_days > 0:
                metrics.position_turnover = metrics.total_trades / trading_days

        # AI指标
        if ai_signals:
            ai_scores = [signal.get("ai_score", 0) for signal in ai_signals]
            confidences = [signal.get("confidence", 0) for signal in ai_signals]

            metrics.avg_ai_score = np.mean(ai_scores) if ai_scores else 0

            # AI评分分布
            score_ranges = [(-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5)]
            for low, high in score_ranges:
                count = sum(1 for score in ai_scores if low <= score < high)
                range_key = f"{low}to{high}"
                metrics.ai_score_distribution[range_key] = count

            # 置信度分布
            conf_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
            for low, high in conf_ranges:
                count = sum(1 for conf in confidences if low <= conf < high)
                range_key = f"{low}to{high}"
                metrics.confidence_distribution[range_key] = count

        return metrics

    def analyze_performance_trend(self, window_size: int = 20) -> Dict:
        """分析性能趋势"""
        if len(self.metrics_history) < window_size:
            return {"error": "数据不足，无法分析趋势"}

        recent_metrics = self.metrics_history[-window_size:]

        # 计算趋势
        trend_analysis = {}

        # 收益趋势
        returns_trend = [m.annual_return for m in recent_metrics]
        trend_analysis["return_trend"] = self._calculate_trend(returns_trend)

        # 风险趋势
        drawdowns_trend = [m.max_drawdown for m in recent_metrics]
        trend_analysis["risk_trend"] = self._calculate_trend(drawdowns_trend)

        # 交易质量趋势
        sharpe_trend = [m.sharpe_ratio for m in recent_metrics]
        trend_analysis["quality_trend"] = self._calculate_trend(sharpe_trend)

        # 稳定性分析
        trend_analysis["stability"] = self._analyze_stability(recent_metrics)

        return trend_analysis

    def _calculate_trend(self, data: List[float]) -> Dict:
        """计算数据趋势"""
        if len(data) < 2:
            return {"direction": "unknown", "slope": 0, "strength": 0}

        x = np.arange(len(data))
        y = np.array(data)

        # 线性回归
        slope = np.polyfit(x, y, 1)[0]

        # 趋势方向
        if slope > 0.001:
            direction = "upward"
        elif slope < -0.001:
            direction = "downward"
        else:
            direction = "stable"

        # 趋势强度（基于R²）
        correlation = np.corrcoef(x, y)[0, 1]
        strength = correlation**2 if not np.isnan(correlation) else 0

        return {
            "direction": direction,
            "slope": slope,
            "strength": strength,
            "correlation": correlation,
        }

    def _analyze_stability(self, metrics_list: List[PerformanceMetrics]) -> Dict:
        """分析策略稳定性"""
        if len(metrics_list) < 5:
            return {"stability_score": 0, "analysis": "数据不足"}

        # 计算各指标的变异系数
        returns = [m.annual_return for m in metrics_list]
        drawdowns = [m.max_drawdown for m in metrics_list]
        sharpe_ratios = [m.sharpe_ratio for m in metrics_list]

        stability_score = 0
        analysis_parts = []

        # 收益稳定性
        if np.std(returns) > 0:
            return_cv = np.std(returns) / abs(np.mean(returns))
            if return_cv < 0.5:
                stability_score += 30
                analysis_parts.append("收益稳定性良好")
            else:
                analysis_parts.append("收益波动较大")

        # 风险控制稳定性
        if np.mean(drawdowns) > 0:
            drawdown_cv = np.std(drawdowns) / np.mean(drawdowns)
            if drawdown_cv < 0.3:
                stability_score += 30
                analysis_parts.append("风险控制稳定")
            else:
                analysis_parts.append("风险控制波动较大")

        # 夏普比率稳定性
        if np.std(sharpe_ratios) > 0:
            sharpe_cv = np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios))
            if sharpe_cv < 0.4:
                stability_score += 40
                analysis_parts.append("风险调整收益稳定")
            else:
                analysis_parts.append("风险调整收益波动较大")

        return {
            "stability_score": stability_score,
            "analysis": "; ".join(analysis_parts),
            "return_cv": np.std(returns) / abs(np.mean(returns))
            if returns and np.mean(returns) != 0
            else 0,
            "drawdown_cv": np.std(drawdowns) / np.mean(drawdowns)
            if drawdowns and np.mean(drawdowns) != 0
            else 0,
            "sharpe_cv": np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios))
            if sharpe_ratios and np.mean(sharpe_ratios) != 0
            else 0,
        }

    def generate_performance_report(self) -> Dict:
        """生成性能报告"""
        if not self.current_metrics:
            return {"error": "没有性能数据"}

        thresholds = self.analysis_config["performance_thresholds"]

        # 评级系统
        rating = self._calculate_performance_rating()

        # 风险评估
        risk_assessment = self._assess_risk_level()

        # 优化建议
        recommendations = self._generate_optimization_recommendations()

        # 趋势分析
        trend_analysis = self.analyze_performance_trend()

        report = {
            "summary": {
                "overall_rating": rating,
                "risk_level": risk_assessment["level"],
                "performance_score": self._calculate_overall_score(),
                "analysis_date": datetime.now().isoformat(),
            },
            "current_metrics": asdict(self.current_metrics),
            "performance_rating": rating,
            "risk_assessment": risk_assessment,
            "trend_analysis": trend_analysis,
            "recommendations": recommendations,
            "benchmark_comparison": self._compare_with_benchmark(),
        }

        return report

    def _calculate_performance_rating(self) -> Dict:
        """计算性能评级"""
        thresholds = self.analysis_config["performance_thresholds"]
        metrics = self.current_metrics

        rating_score = 0
        rating_details = []

        # 夏普比率评级
        if metrics.sharpe_ratio >= thresholds["excellent_sharpe"]:
            rating_score += 30
            rating_details.append("夏普比率: 优秀")
        elif metrics.sharpe_ratio >= thresholds["good_sharpe"]:
            rating_score += 20
            rating_details.append("夏普比率: 良好")
        else:
            rating_details.append("夏普比率: 需改进")

        # 胜率评级
        if metrics.win_rate >= thresholds["excellent_win_rate"]:
            rating_score += 25
            rating_details.append("胜率: 优秀")
        elif metrics.win_rate >= thresholds["good_win_rate"]:
            rating_score += 15
            rating_details.append("胜率: 良好")
        else:
            rating_details.append("胜率: 需改进")

        # 回撤控制评级
        if metrics.max_drawdown <= thresholds["max_acceptable_drawdown"]:
            rating_score += 25
            rating_details.append("回撤控制: 优秀")
        else:
            rating_details.append("回撤控制: 需改进")

        # 收益评级
        if metrics.annual_return >= thresholds["min_acceptable_return"]:
            rating_score += 20
            rating_details.append("年化收益: 优秀")
        else:
            rating_details.append("年化收益: 需改进")

        # 总体评级
        if rating_score >= 80:
            overall_rating = "A+ (优秀)"
        elif rating_score >= 70:
            overall_rating = "A (良好)"
        elif rating_score >= 60:
            overall_rating = "B (中等)"
        elif rating_score >= 50:
            overall_rating = "C (一般)"
        else:
            overall_rating = "D (需改进)"

        return {
            "overall": overall_rating,
            "score": rating_score,
            "details": rating_details,
        }

    def _assess_risk_level(self) -> Dict:
        """评估风险水平"""
        risk_limits = self.analysis_config["risk_limits"]
        metrics = self.current_metrics

        risk_factors = []
        risk_score = 0

        # 最大回撤风险
        if metrics.max_drawdown > risk_limits["max_drawdown"]:
            risk_score += 40
            risk_factors.append(
                f"最大回撤超限: {metrics.max_drawdown:.1%} > {risk_limits['max_drawdown']:.1%}"
            )

        # 波动率风险
        if metrics.volatility > 0.3:
            risk_score += 30
            risk_factors.append(f"波动率过高: {metrics.volatility:.1%}")

        # 夏普比率风险
        if metrics.sharpe_ratio < 0.5:
            risk_score += 20
            risk_factors.append(f"风险调整收益过低: {metrics.sharpe_ratio:.2f}")

        # 胜率风险
        if metrics.win_rate < 0.4:
            risk_score += 10
            risk_factors.append(f"胜率过低: {metrics.win_rate:.1%}")

        # 风险等级
        if risk_score >= 70:
            risk_level = "高风险"
        elif risk_score >= 40:
            risk_level = "中等风险"
        else:
            risk_level = "低风险"

        return {"level": risk_level, "score": risk_score, "factors": risk_factors}

    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        metrics = self.current_metrics
        thresholds = self.analysis_config["performance_thresholds"]

        # 收益优化建议
        if metrics.annual_return < thresholds["min_acceptable_return"]:
            recommendations.append("考虑提高AI评分阈值，选择更高质量的信号")
            recommendations.append("优化仓位管理，适当提高高置信度信号的仓位")

        # 风险控制建议
        if metrics.max_drawdown > thresholds["max_acceptable_drawdown"]:
            recommendations.append("加强止损管理，降低单笔最大损失")
            recommendations.append("考虑降低最大持仓数量，减少组合风险")

        # 交易质量建议
        if metrics.sharpe_ratio < thresholds["good_sharpe"]:
            recommendations.append("优化信号过滤条件，提高信号质量")
            recommendations.append("考虑增加市场环境判断，避免不利时期交易")

        # 胜率优化建议
        if metrics.win_rate < thresholds["good_win_rate"]:
            recommendations.append("提高置信度要求，减少低质量交易")
            recommendations.append("优化因子权重，提升预测准确性")

        # 持仓管理建议
        if metrics.avg_holding_period > 10:
            recommendations.append("考虑缩短持仓周期，提高资金利用效率")
        elif metrics.avg_holding_period < 3:
            recommendations.append("适当延长持仓周期，避免过度交易")

        # AI模型优化建议
        if metrics.avg_ai_score < 1.0:
            recommendations.append("AI评分偏低，考虑重新训练或调整模型参数")

        return recommendations

    def _compare_with_benchmark(self) -> Dict:
        """与基准比较"""
        if not self.benchmark_metrics:
            return {"error": "没有基准数据"}

        comparison = {}

        # 收益比较
        return_diff = (
            self.current_metrics.annual_return - self.benchmark_metrics.annual_return
        )
        comparison["return_outperformance"] = return_diff

        # 风险比较
        drawdown_diff = (
            self.current_metrics.max_drawdown - self.benchmark_metrics.max_drawdown
        )
        comparison["drawdown_comparison"] = drawdown_diff

        # 风险调整收益比较
        sharpe_diff = (
            self.current_metrics.sharpe_ratio - self.benchmark_metrics.sharpe_ratio
        )
        comparison["sharpe_outperformance"] = sharpe_diff

        # 总体评估
        outperformance_score = 0
        if return_diff > 0:
            outperformance_score += 30
        if drawdown_diff < 0:
            outperformance_score += 20
        if sharpe_diff > 0:
            outperformance_score += 50

        comparison["overall_outperformance"] = outperformance_score

        return comparison

    def _calculate_overall_score(self) -> float:
        """计算总体得分"""
        metrics = self.current_metrics

        # 收益得分 (30%)
        return_score = min(100, max(0, metrics.annual_return * 100)) * 0.3

        # 风险得分 (25%)
        risk_score = min(100, max(0, (1 - metrics.max_drawdown) * 100)) * 0.25

        # 夏普比率得分 (25%)
        sharpe_score = min(100, max(0, metrics.sharpe_ratio * 25)) * 0.25

        # 胜率得分 (20%)
        win_score = min(100, max(0, metrics.win_rate * 100)) * 0.2

        total_score = return_score + risk_score + sharpe_score + win_score

        return total_score

    def create_performance_dashboard(
        self, save_path: str = "performance_dashboard.png"
    ):
        """创建性能仪表板"""
        if not self.current_metrics:
            print("没有性能数据可显示")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("AI智能体策略性能仪表板", fontsize=16, fontweight="bold")

        # 1. 收益曲线
        ax1 = axes[0, 0]
        if hasattr(self, "equity_curve"):
            ax1.plot(self.equity_curve, linewidth=2, color="blue")
            ax1.set_title("资金曲线")
            ax1.set_ylabel("账户价值")
            ax1.grid(True, alpha=0.3)

        # 2. 关键指标
        ax2 = axes[0, 1]
        metrics_data = [
            self.current_metrics.annual_return * 100,
            self.current_metrics.sharpe_ratio,
            self.current_metrics.win_rate * 100,
            (1 - self.current_metrics.max_drawdown) * 100,
        ]
        metric_labels = ["年化收益", "夏普比率", "胜率", "回撤控制"]
        colors = ["green", "blue", "orange", "red"]

        bars = ax2.bar(metric_labels, metrics_data, color=colors, alpha=0.7)
        ax2.set_title("关键性能指标")
        ax2.set_ylabel("数值")

        # 添加数值标签
        for bar, value in zip(bars, metrics_data):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        # 3. AI评分分布
        ax3 = axes[0, 2]
        if self.current_metrics.ai_score_distribution:
            score_ranges = list(self.current_metrics.ai_score_distribution.keys())
            score_counts = list(self.current_metrics.ai_score_distribution.values())

            ax3.bar(score_ranges, score_counts, color="purple", alpha=0.7)
            ax3.set_title("AI评分分布")
            ax3.set_xlabel("评分范围")
            ax3.set_ylabel("频次")
            ax3.tick_params(axis="x", rotation=45)

        # 4. 置信度分布
        ax4 = axes[1, 0]
        if self.current_metrics.confidence_distribution:
            conf_ranges = list(self.current_metrics.confidence_distribution.keys())
            conf_counts = list(self.current_metrics.confidence_distribution.values())

            ax4.bar(conf_ranges, conf_counts, color="cyan", alpha=0.7)
            ax4.set_title("置信度分布")
            ax4.set_xlabel("置信度范围")
            ax4.set_ylabel("频次")
            ax4.tick_params(axis="x", rotation=45)

        # 5. 月度收益热力图
        ax5 = axes[1, 1]
        if hasattr(self, "monthly_returns"):
            monthly_data = self.monthly_returns
            if monthly_data:
                sns.heatmap(
                    monthly_data.reshape(-1, 1)
                    if len(monthly_data.shape) == 1
                    else monthly_data,
                    annot=True,
                    cmap="RdYlGn",
                    center=0,
                    ax=ax5,
                )
                ax5.set_title("月度收益热力图")

        # 6. 风险收益散点图
        ax6 = axes[1, 2]
        if len(self.metrics_history) > 1:
            returns = [m.annual_return for m in self.metrics_history]
            drawdowns = [m.max_drawdown for m in self.metrics_history]

            ax6.scatter(drawdowns, returns, alpha=0.6, s=50)
            ax6.scatter(
                self.current_metrics.max_drawdown,
                self.current_metrics.annual_return,
                color="red",
                s=100,
                marker="*",
                label="当前",
            )
            ax6.set_title("风险收益分布")
            ax6.set_xlabel("最大回撤")
            ax6.set_ylabel("年化收益")
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"性能仪表板已保存到: {save_path}")

    def save_performance_data(self, filename: str = "performance_data.json"):
        """保存性能数据"""
        data = {
            "current_metrics": asdict(self.current_metrics),
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "benchmark_metrics": asdict(self.benchmark_metrics),
            "analysis_config": self.analysis_config,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"性能数据已保存到: {filename}")

    def load_performance_data(self, filename: str = "performance_data.json"):
        """加载性能数据"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.current_metrics = PerformanceMetrics(**data["current_metrics"])
            self.metrics_history = [
                PerformanceMetrics(**m) for m in data["metrics_history"]
            ]
            self.benchmark_metrics = PerformanceMetrics(**data["benchmark_metrics"])

            if "analysis_config" in data:
                self.analysis_config.update(data["analysis_config"])

            print(f"已加载性能数据，历史记录数: {len(self.metrics_history)}")
            return True
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return False


def main():
    """主函数"""
    analyzer = PerformanceAnalyzer()

    print("=" * 60)
    print("AI智能体策略性能监控与分析系统")
    print("=" * 60)

    # 模拟数据生成（实际使用时应该从真实数据源获取）
    print("\n正在生成模拟性能数据...")

    # 模拟资金曲线
    days = 252
    initial_value = 1000000
    daily_return = 0.0015  # 日均收益0.15%
    volatility = 0.02  # 日波动率2%

    np.random.seed(42)
    daily_returns = np.random.normal(daily_return, volatility, days)
    equity_curve = initial_value * (1 + np.cumsum(daily_returns))

    # 模拟交易数据
    trades = []
    for i in range(50):  # 50笔交易
        trade_return = np.random.normal(0.02, 0.05)  # 平均2%收益
        trades.append(
            {
                "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "return": trade_return,
                "holding_days": np.random.randint(1, 15),
            }
        )

    # 模拟持仓数据
    positions = []
    for i in range(days):
        positions.append(
            {
                "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "position_count": np.random.randint(10, 25),
                "holding_days": np.random.randint(1, 10),
            }
        )

    # 模拟AI信号数据
    ai_signals = []
    for i in range(100):
        ai_signals.append(
            {
                "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "ai_score": np.random.uniform(-3, 3),
                "confidence": np.random.uniform(0.4, 0.9),
            }
        )

    # 计算性能指标
    print("正在计算性能指标...")
    analyzer.current_metrics = analyzer.calculate_performance_metrics(
        equity_curve, trades, positions, ai_signals
    )

    # 生成性能报告
    print("\n正在生成性能报告...")
    report = analyzer.generate_performance_report()

    # 显示报告
    print("\n" + "=" * 60)
    print("性能分析报告")
    print("=" * 60)

    print(f"\n总体评级: {report['performance_rating']['overall']}")
    print(f"风险等级: {report['risk_assessment']['level']}")
    print(f"总体得分: {report['summary']['performance_score']:.1f}/100")

    print(f"\n关键指标:")
    print(f"  年化收益: {analyzer.current_metrics.annual_return:.2%}")
    print(f"  夏普比率: {analyzer.current_metrics.sharpe_ratio:.2f}")
    print(f"  最大回撤: {analyzer.current_metrics.max_drawdown:.2%}")
    print(f"  胜率: {analyzer.current_metrics.win_rate:.2%}")
    print(f"  总交易次数: {analyzer.current_metrics.total_trades}")

    print(f"\n优化建议:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")

    # 创建性能仪表板
    print("\n正在创建性能仪表板...")
    analyzer.equity_curve = equity_curve
    analyzer.create_performance_dashboard()

    # 保存数据
    analyzer.save_performance_data()

    print("\n性能分析完成! 请查看生成的报告和图表。")


if __name__ == "__main__":
    main()
