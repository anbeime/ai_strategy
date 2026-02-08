#!/usr/bin/python
"""
AI智能体策略集成系统
整合参数优化、性能监控和策略执行，提供完整的AI策略解决方案
"""

import numpy as np
import pandas as pd
import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings("ignore")

# 导入自定义模块
try:
    from parameter_optimizer import ParameterOptimizer
    from performance_monitor import PerformanceAnalyzer

    print("✓ 成功导入参数优化和性能监控模块")
except ImportError as e:
    print(f"⚠ 模块导入警告: {e}")
    print("将使用内置简化版本")


class AIStrategyIntegration:
    """AI策略集成系统"""

    def __init__(self, config_file: str = "ai_strategy_config.json"):
        self.config_file = config_file
        self.config = self.load_config()

        # 初始化组件
        self.optimizer = None
        self.performance_analyzer = None
        self.strategy_state = {}

        # 数据存储
        self.historical_data = {}
        self.performance_history = []
        self.optimization_results = []

        # 状态标志
        self.is_optimized = False
        self.is_monitoring = False

        print("AI智能体策略集成系统初始化完成")

    def load_config(self) -> Dict:
        """加载配置文件"""
        default_config = {
            "strategy": {
                "name": "AI智能体策略",
                "version": "1.0.0",
                "description": "基于北京大学光华管理学院研究的AI智能体投资策略",
            },
            "parameters": {
                "min_ai_score": 1.5,
                "min_confidence": 0.6,
                "max_positions": 20,
                "position_size": 0.05,
                "stop_loss": 0.10,
                "profit_target": 0.15,
                "rebalance_frequency": "daily",
            },
            "optimization": {
                "enabled": True,
                "mode": "grid",
                "max_iterations": 50,
                "optimization_frequency": "weekly",
            },
            "monitoring": {
                "enabled": True,
                "real_time_alerts": True,
                "performance_dashboard": True,
                "risk_monitoring": True,
            },
            "data": {
                "source": "qmt",
                "stock_pool_file": "C:/F/stock_pool_2509.txt",
                "backup_data_source": "tdx",
            },
            "risk_management": {
                "max_drawdown": 0.15,
                "max_daily_loss": 0.05,
                "position_concentration": 0.10,
                "sector_exposure": 0.30,
            },
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                # 合并默认配置
                config = {**default_config, **loaded_config}
                print(f"✓ 已加载配置文件: {self.config_file}")
            else:
                config = default_config
                self.save_config(config)
                print(f"✓ 创建默认配置文件: {self.config_file}")
        except Exception as e:
            print(f"⚠ 配置文件加载失败，使用默认配置: {e}")
            config = default_config

        return config

    def save_config(self, config: Dict = None):
        """保存配置文件"""
        if config is None:
            config = self.config

        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✓ 配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"⚠ 配置保存失败: {e}")

    def initialize_components(self):
        """初始化各个组件"""
        print("\n正在初始化系统组件...")

        # 初始化参数优化器
        try:
            if "ParameterOptimizer" in globals():
                self.optimizer = ParameterOptimizer()
                print("✓ 参数优化器初始化完成")
            else:
                print("⚠ 参数优化器不可用，将跳过优化功能")
        except Exception as e:
            print(f"⚠ 参数优化器初始化失败: {e}")

        # 初始化性能监控器
        try:
            if "PerformanceAnalyzer" in globals():
                self.performance_analyzer = PerformanceAnalyzer()
                print("✓ 性能监控器初始化完成")
            else:
                print("⚠ 性能监控器不可用，将跳过监控功能")
        except Exception as e:
            print(f"⚠ 性能监控器初始化失败: {e}")

        # 初始化策略状态
        self.strategy_state = {
            "current_positions": {},
            "cash_balance": 1000000.0,
            "total_value": 1000000.0,
            "daily_pnl": 0.0,
            "last_update": datetime.now().isoformat(),
            "signals_today": [],
            "trades_today": [],
        }

        print("✓ 策略状态初始化完成")

    def run_parameter_optimization(self) -> Dict:
        """运行参数优化"""
        if not self.optimizer:
            return {"error": "参数优化器未初始化"}

        print("\n开始参数优化...")

        # 获取优化配置
        opt_config = self.config["optimization"]
        mode = opt_config.get("mode", "grid")
        max_iterations = opt_config.get("max_iterations", 50)

        # 运行优化
        try:
            optimization_report = self.optimizer.run_optimization(
                mode=mode, max_iterations=max_iterations
            )

            # 保存优化结果
            self.optimization_results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "report": optimization_report,
                    "config_used": self.config["parameters"],
                }
            )

            # 更新最佳参数
            if optimization_report.get("best_parameters"):
                self.config["parameters"].update(optimization_report["best_parameters"])
                self.save_config()
                self.is_optimized = True
                print("✓ 参数已更新到配置文件")

            return optimization_report

        except Exception as e:
            print(f"⚠ 参数优化失败: {e}")
            return {"error": str(e)}

    def start_performance_monitoring(self):
        """启动性能监控"""
        if not self.performance_analyzer:
            print("⚠ 性能监控器未初始化，无法启动监控")
            return

        print("\n启动性能监控...")
        self.is_monitoring = True

        # 模拟监控数据（实际使用时应该从真实数据源获取）
        self._simulate_monitoring_data()

        print("✓ 性能监控已启动")

    def _simulate_monitoring_data(self):
        """模拟监控数据"""
        # 生成模拟数据
        days = 60
        initial_value = 1000000

        np.random.seed(42)
        daily_returns = np.random.normal(0.0015, 0.02, days)
        equity_curve = initial_value * (1 + np.cumsum(daily_returns))

        # 模拟交易数据
        trades = []
        for i in range(30):
            trades.append(
                {
                    "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                    "return": np.random.normal(0.02, 0.05),
                    "holding_days": np.random.randint(1, 10),
                }
            )

        # 模拟持仓数据
        positions = []
        for i in range(days):
            positions.append(
                {
                    "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                    "position_count": np.random.randint(15, 25),
                    "holding_days": np.random.randint(1, 8),
                }
            )

        # 模拟AI信号数据
        ai_signals = []
        for i in range(80):
            ai_signals.append(
                {
                    "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                    "ai_score": np.random.uniform(-2, 3),
                    "confidence": np.random.uniform(0.5, 0.9),
                }
            )

        # 计算性能指标
        metrics = self.performance_analyzer.calculate_performance_metrics(
            equity_curve.tolist(), trades, positions, ai_signals
        )

        self.performance_analyzer.current_metrics = metrics
        self.performance_analyzer.equity_curve = equity_curve.tolist()

        print(f"✓ 模拟监控数据生成完成")
        print(f"  - 总收益率: {metrics.total_return:.2%}")
        print(f"  - 夏普比率: {metrics.sharpe_ratio:.2f}")
        print(f"  - 最大回撤: {metrics.max_drawdown:.2%}")
        print(f"  - 胜率: {metrics.win_rate:.2%}")

    def generate_comprehensive_report(self) -> Dict:
        """生成综合报告"""
        print("\n生成综合报告...")

        report = {
            "report_info": {
                "generated_at": datetime.now().isoformat(),
                "strategy_name": self.config["strategy"]["name"],
                "version": self.config["strategy"]["version"],
            },
            "strategy_config": self.config,
            "optimization_status": {
                "is_optimized": self.is_optimized,
                "last_optimization": self.optimization_results[-1]["timestamp"]
                if self.optimization_results
                else None,
                "optimization_count": len(self.optimization_results),
            },
            "current_performance": {},
            "risk_assessment": {},
            "recommendations": [],
            "next_steps": [],
        }

        # 添加性能分析
        if self.performance_analyzer and self.performance_analyzer.current_metrics:
            performance_report = self.performance_analyzer.generate_performance_report()
            report["current_performance"] = performance_report.get(
                "current_metrics", {}
            )
            report["risk_assessment"] = performance_report.get("risk_assessment", {})
            report["recommendations"] = performance_report.get("recommendations", [])

        # 添加优化结果
        if self.optimization_results:
            latest_optimization = self.optimization_results[-1]
            report["latest_optimization"] = latest_optimization["report"]

        # 生成下一步建议
        report["next_steps"] = self._generate_next_steps()

        # 保存报告
        report_filename = f"ai_strategy_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✓ 综合报告已保存到: {report_filename}")
        return report

    def _generate_next_steps(self) -> List[str]:
        """生成下一步建议"""
        next_steps = []

        # 基于优化状态的建议
        if not self.is_optimized:
            next_steps.append("运行参数优化以提升策略性能")
        else:
            next_steps.append("定期重新运行参数优化以适应市场变化")

        # 基于监控状态的建议
        if not self.is_monitoring:
            next_steps.append("启动实时性能监控")

        # 基于性能的建议
        if self.performance_analyzer and self.performance_analyzer.current_metrics:
            metrics = self.performance_analyzer.current_metrics

            if metrics.sharpe_ratio < 1.0:
                next_steps.append("优化因子权重以提高风险调整收益")

            if metrics.max_drawdown > 0.15:
                next_steps.append("加强风险管理，降低最大回撤")

            if metrics.win_rate < 0.5:
                next_steps.append("提高信号质量标准，改善胜率")

        # 基于配置的建议
        if self.config["optimization"]["enabled"]:
            next_steps.append("设置自动优化计划，定期更新策略参数")

        if self.config["monitoring"]["enabled"]:
            next_steps.append("配置实时告警，及时响应策略异常")

        # 实施建议
        next_steps.append("在QMT系统中测试优化后的策略参数")
        next_steps.append("考虑实盘小额资金验证策略效果")

        return next_steps

    def create_strategy_dashboard(self):
        """创建策略仪表板"""
        print("\n创建策略仪表板...")

        if not self.performance_analyzer:
            print("⚠ 性能监控器未初始化，无法创建仪表板")
            return

        try:
            # 创建性能仪表板
            dashboard_path = "ai_strategy_dashboard.png"
            self.performance_analyzer.create_performance_dashboard(dashboard_path)

            # 创建优化结果图表
            if self.optimization_results:
                self._create_optimization_chart()

            print("✓ 策略仪表板创建完成")

        except Exception as e:
            print(f"⚠ 仪表板创建失败: {e}")

    def _create_optimization_chart(self):
        """创建优化结果图表"""
        try:
            import matplotlib.pyplot as plt

            # 提取优化数据
            scores = []
            params = []

            for result in self.optimization_results:
                if "report" in result and "summary" in result["report"]:
                    scores.append(result["report"]["summary"]["best_score"])
                    params.append(result["report"]["best_parameters"])

            if scores:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # 优化得分趋势
                ax1.plot(range(len(scores)), scores, marker="o", linewidth=2)
                ax1.set_title("参数优化得分趋势")
                ax1.set_xlabel("优化次数")
                ax1.set_ylabel("最佳得分")
                ax1.grid(True, alpha=0.3)

                # 参数变化（选择关键参数）
                if params:
                    key_params = ["min_ai_score", "min_confidence", "max_positions"]
                    for param in key_params:
                        if param in params[0]:
                            values = [p.get(param, 0) for p in params]
                            ax2.plot(
                                range(len(values)), values, marker="s", label=param
                            )

                    ax2.set_title("关键参数变化趋势")
                    ax2.set_xlabel("优化次数")
                    ax2.set_ylabel("参数值")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(
                    "ai_strategy_optimization_trend.png", dpi=300, bbox_inches="tight"
                )
                plt.show()

                print("✓ 优化趋势图表已保存")

        except ImportError:
            print("⚠ matplotlib不可用，无法创建图表")
        except Exception as e:
            print(f"⚠ 图表创建失败: {e}")

    def run_full_workflow(self):
        """运行完整工作流程"""
        print("=" * 80)
        print("AI智能体策略完整工作流程")
        print("=" * 80)

        start_time = time.time()

        # 1. 初始化组件
        self.initialize_components()

        # 2. 参数优化
        if self.config["optimization"]["enabled"]:
            optimization_result = self.run_parameter_optimization()
            if "error" not in optimization_result:
                print("✓ 参数优化完成")
            else:
                print("⚠ 参数优化失败，继续使用默认参数")

        # 3. 启动性能监控
        if self.config["monitoring"]["enabled"]:
            self.start_performance_monitoring()

        # 4. 生成综合报告
        comprehensive_report = self.generate_comprehensive_report()

        # 5. 创建仪表板
        if self.config["monitoring"]["performance_dashboard"]:
            self.create_strategy_dashboard()

        # 6. 显示总结
        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 80)
        print("工作流程完成总结")
        print("=" * 80)
        print(f"总耗时: {duration:.2f}秒")
        print(f"策略名称: {self.config['strategy']['name']}")
        print(f"版本: {self.config['strategy']['version']}")

        if self.is_optimized:
            print("✓ 参数优化: 已完成")
        else:
            print("⚠ 参数优化: 未完成")

        if self.is_monitoring:
            print("✓ 性能监控: 已启动")
        else:
            print("⚠ 性能监控: 未启动")

        # 显示关键指标
        if self.performance_analyzer and self.performance_analyzer.current_metrics:
            metrics = self.performance_analyzer.current_metrics
            print(f"\n关键性能指标:")
            print(f"  总收益率: {metrics.total_return:.2%}")
            print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
            print(f"  最大回撤: {metrics.max_drawdown:.2%}")
            print(f"  胜率: {metrics.win_rate:.2%}")

        # 显示下一步建议
        print(f"\n下一步建议:")
        for i, step in enumerate(comprehensive_report["next_steps"][:5], 1):
            print(f"  {i}. {step}")

        print(f"\n📊 详细报告和图表已生成，请查看相关文件")
        print("=" * 80)

    def save_system_state(self):
        """保存系统状态"""
        state = {
            "config": self.config,
            "strategy_state": self.strategy_state,
            "is_optimized": self.is_optimized,
            "is_monitoring": self.is_monitoring,
            "optimization_results": self.optimization_results,
            "performance_history": self.performance_history,
            "timestamp": datetime.now().isoformat(),
        }

        state_file = "ai_strategy_system_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        print(f"✓ 系统状态已保存到: {state_file}")

    def load_system_state(self):
        """加载系统状态"""
        state_file = "ai_strategy_system_state.json"

        try:
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)

                self.config = state.get("config", self.config)
                self.strategy_state = state.get("strategy_state", {})
                self.is_optimized = state.get("is_optimized", False)
                self.is_monitoring = state.get("is_monitoring", False)
                self.optimization_results = state.get("optimization_results", [])
                self.performance_history = state.get("performance_history", [])

                print(f"✓ 系统状态已从 {state_file} 恢复")
                return True
        except Exception as e:
            print(f"⚠ 系统状态恢复失败: {e}")

        return False


def main():
    """主函数"""
    print("AI智能体策略集成系统")
    print("基于北京大学光华管理学院前沿研究")
    print("=" * 80)

    # 创建集成系统
    integration = AIStrategyIntegration()

    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 完整工作流程 (推荐)")
    print("2. 仅参数优化")
    print("3. 仅性能监控")
    print("4. 生成综合报告")
    print("5. 恢复系统状态并继续")

    choice = input("请输入选择 (1-5): ").strip()

    try:
        if choice == "1":
            integration.run_full_workflow()
        elif choice == "2":
            integration.initialize_components()
            integration.run_parameter_optimization()
        elif choice == "3":
            integration.initialize_components()
            integration.start_performance_monitoring()
            integration.create_strategy_dashboard()
        elif choice == "4":
            integration.initialize_components()
            integration.start_performance_monitoring()
            integration.generate_comprehensive_report()
        elif choice == "5":
            if integration.load_system_state():
                integration.run_full_workflow()
            else:
                print("系统状态恢复失败，运行完整工作流程")
                integration.run_full_workflow()
        else:
            print("无效选择，运行完整工作流程")
            integration.run_full_workflow()

        # 保存系统状态
        integration.save_system_state()

    except KeyboardInterrupt:
        print("\n\n用户中断，正在保存系统状态...")
        integration.save_system_state()
        print("系统状态已保存，程序退出")
    except Exception as e:
        print(f"\n⚠ 运行过程中发生错误: {e}")
        print("正在保存系统状态...")
        integration.save_system_state()

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
