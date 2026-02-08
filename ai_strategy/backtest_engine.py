#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI智能体回测引擎
集成通达信数据，实现完整的策略回测功能
"""

import os
import json
import asyncio
import numpy as np
import pandas as pd
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# 导入自定义模块
from information_processor import InformationCollector, InformationProcessor, MarketInfo
from signal_decision_system import (
    SignalGenerator,
    DecisionEngine,
    TradingExecutor,
    TradingSignal,
    PortfolioDecision,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测结果"""

    start_date: str
    end_date: str
    initial_cash: float
    final_cash: float
    final_assets: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_trade_return: float
    volatility: float
    daily_returns: List[float]
    equity_curve: List[Dict[str, Any]]
    trade_history: List[Dict[str, Any]]


@dataclass
class DailyResult:
    """每日结果"""

    date: str
    cash: float
    positions: Dict[str, int]
    position_values: Dict[str, float]
    total_assets: float
    daily_return: float
    trades: List[Dict[str, Any]]
    signals: List[Dict[str, Any]]


class TDXDataReader:
    """通达信数据读取器"""

    def __init__(self, tdx_path: str = "C:/F/newtdx"):
        """
        初始化通达信数据读取器

        Args:
            tdx_path: 通达信数据路径
        """
        self.tdx_path = Path(tdx_path)
        self.vipdoc_path = self.tdx_path / "vipdoc"
        logger.info(f"通达信数据读取器初始化: {tdx_path}")

    def get_day_file_path(self, stock_code: str) -> Optional[Path]:
        """
        获取股票日线文件路径

        Args:
            stock_code: 股票代码

        Returns:
            文件路径
        """
        stock_code = str(stock_code)

        if (
            stock_code.startswith("00")
            or stock_code.startswith("30")
            or stock_code.startswith("1")
            or stock_code.startswith("39")
        ):
            return self.vipdoc_path / "sz" / "lday" / f"sz{stock_code}.day"
        elif stock_code.startswith("6") or stock_code.startswith("5"):
            return self.vipdoc_path / "sh" / "lday" / f"sh{stock_code}.day"
        elif stock_code.startswith("68") or stock_code.startswith("43"):
            prefix = "sh" if stock_code.startswith("68") else "bj"
            return self.vipdoc_path / prefix / "lday" / f"{prefix}{stock_code}.day"
        else:
            return None

    def read_day_data(
        self, stock_code: str, start_date: datetime = None, end_date: datetime = None
    ) -> Optional[pd.DataFrame]:
        """
        读取日线数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票数据DataFrame
        """
        file_path = self.get_day_file_path(stock_code)

        if not file_path or not file_path.exists():
            logger.warning(f"数据文件不存在: {file_path}")
            return None

        try:
            data = []
            with open(file_path, "rb") as f:
                while True:
                    buffer = f.read(32)
                    if len(buffer) < 32:
                        break

                    record = struct.unpack("<IIIIIfII", buffer)
                    date_int = record[0]
                    open_price = record[1] / 100.0
                    high_price = record[2] / 100.0
                    low_price = record[3] / 100.0
                    close_price = record[4] / 100.0
                    volume = record[5]

                    try:
                        date = datetime.strptime(str(date_int), "%Y%m%d")
                    except ValueError:
                        continue

                    if start_date and date.date() < start_date.date():
                        continue
                    if end_date and date.date() > end_date.date():
                        break

                    data.append(
                        {
                            "date": date,
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": volume,
                        }
                    )

            if data:
                df = pd.DataFrame(data)
                df.set_index("date", inplace=True)
                return df
            else:
                logger.warning(f"没有找到{stock_code}的有效数据")
                return None

        except Exception as e:
            logger.error(f"读取{stock_code}数据失败: {e}")
            return None

    def get_price_data(
        self, stock_code: str, date: datetime
    ) -> Optional[Dict[str, float]]:
        """
        获取指定日期的价格数据

        Args:
            stock_code: 股票代码
            date: 查询日期

        Returns:
            价格数据
        """
        df = self.read_day_data(stock_code, date, date)

        if df is not None and len(df) > 0:
            row = df.iloc[0]
            return {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }

        return None

    def load_stock_pool_data(
        self, stock_codes: List[str], start_date: datetime, end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载股票池数据

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票数据字典
        """
        logger.info(
            f"开始加载{len(stock_codes)}只股票的数据，时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
        )

        stock_data = {}

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                logger.info(f"  已加载 {i + 1}/{len(stock_codes)} 只股票")

            df = self.read_day_data(code, start_date, end_date)
            if df is not None and len(df) > 0:
                stock_data[code] = df

        logger.info(f"成功加载 {len(stock_data)} 只股票的数据")
        return stock_data


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化回测引擎

        Args:
            config: 配置信息
        """
        self.config = config or {}

        # 回测参数
        self.initial_cash = self.config.get("initial_cash", 100000.0)
        self.max_positions = self.config.get("max_positions", 20)
        self.commission_rate = self.config.get("commission_rate", 0.0003)

        # 数据读取器
        self.data_reader = TDXDataReader(self.config.get("tdx_path", "C:/F/newtdx"))

        # 策略组件
        self.signal_generator = SignalGenerator(self.config)
        self.decision_engine = DecisionEngine(self.config)
        self.trading_executor = TradingExecutor(self.config)

        # 回测状态
        self.current_cash = self.initial_cash
        self.current_positions = {}
        self.daily_results = []
        self.trade_history = []

        logger.info("回测引擎初始化完成")

    async def run_backtest(
        self, stock_codes: List[str], start_date: datetime, end_date: datetime
    ) -> BacktestResult:
        """
        运行回测

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果
        """
        logger.info(
            f"开始回测: {len(stock_codes)}只股票, {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
        )

        # 重置状态
        self.current_cash = self.initial_cash
        self.current_positions = {}
        self.daily_results = []
        self.trade_history = []

        # 预加载股票数据
        stock_data = self.data_reader.load_stock_pool_data(
            stock_codes, start_date, end_date
        )

        if not stock_data:
            logger.error("没有加载到任何股票数据")
            return self._create_empty_result(start_date, end_date)

        # 生成交易日期列表
        trading_dates = self._generate_trading_dates(start_date, end_date, stock_data)

        logger.info(f"共找到 {len(trading_dates)} 个交易日")

        # 逐日回测
        for i, date in enumerate(trading_dates):
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  回测进度: {i + 1}/{len(trading_dates)} ({date.strftime('%Y-%m-%d')})"
                )

            daily_result = await self._run_single_day(date, stock_codes, stock_data)
            self.daily_results.append(daily_result)

        # 计算最终结果
        backtest_result = self._calculate_backtest_result(start_date, end_date)

        logger.info(
            f"回测完成: 总收益率 {backtest_result.total_return:.2%}, 夏普比率 {backtest_result.sharpe_ratio:.2f}"
        )

        return backtest_result

    def _generate_trading_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        stock_data: Dict[str, pd.DataFrame],
    ) -> List[datetime]:
        """
        生成交易日期列表

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_data: 股票数据

        Returns:
            交易日期列表
        """
        # 获取所有股票的日期并取交集
        all_dates = None

        for df in stock_data.values():
            dates = set(df.index.date)
            if all_dates is None:
                all_dates = dates
            else:
                all_dates = all_dates.intersection(dates)

        if all_dates is None:
            return []

        # 过滤日期范围并排序
        trading_dates = [
            datetime.combine(date, datetime.min.time())
            for date in sorted(all_dates)
            if start_date.date() <= date <= end_date.date()
        ]

        return trading_dates

    async def _run_single_day(
        self,
        date: datetime,
        stock_codes: List[str],
        stock_data: Dict[str, pd.DataFrame],
    ) -> DailyResult:
        """
        运行单日回测

        Args:
            date: 交易日期
            stock_codes: 股票代码列表
            stock_data: 股票数据

        Returns:
            每日结果
        """
        # 1. 生成交易信号
        signals = await self._generate_daily_signals(date, stock_codes, stock_data)

        # 2. 做出投资决策
        decision = self.decision_engine.make_portfolio_decision(
            signals, self.current_positions, self.current_cash, date
        )

        # 3. 执行交易
        execution_result = self.trading_executor.execute_decision(
            decision, self.current_positions, self.current_cash
        )

        # 4. 更新状态
        if execution_result["success"]:
            self.current_cash = execution_result["remaining_cash"]
            self.current_positions = execution_result["executed_positions"]

            # 记录交易历史
            for trade in (
                execution_result["buy_orders"] + execution_result["sell_orders"]
            ):
                if trade["success"]:
                    self.trade_history.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "symbol": trade["symbol"],
                            "action": trade["action"],
                            "quantity": trade["quantity"],
                            "price": trade["price"],
                            "amount": trade.get("cost", trade.get("proceeds", 0)),
                            "timestamp": date,
                        }
                    )

        # 5. 计算当日资产
        total_assets = self._calculate_total_assets(date, stock_data)

        # 6. 计算日收益率
        daily_return = self._calculate_daily_return(total_assets)

        # 7. 计算持仓价值
        position_values = {}
        for symbol, quantity in self.current_positions.items():
            if quantity > 0 and symbol in stock_data:
                price_data = self._get_price_from_data(symbol, date, stock_data[symbol])
                if price_data:
                    position_values[symbol] = quantity * price_data["close"]

        return DailyResult(
            date=date.strftime("%Y-%m-%d"),
            cash=self.current_cash,
            positions=dict(self.current_positions),
            position_values=position_values,
            total_assets=total_assets,
            daily_return=daily_return,
            trades=execution_result["buy_orders"] + execution_result["sell_orders"],
            signals=[asdict(s) for s in signals],
        )

    async def _generate_daily_signals(
        self,
        date: datetime,
        stock_codes: List[str],
        stock_data: Dict[str, pd.DataFrame],
    ) -> List[TradingSignal]:
        """
        生成每日交易信号

        Args:
            date: 交易日期
            stock_codes: 股票代码列表
            stock_data: 股票数据

        Returns:
            交易信号列表
        """
        signals = []

        # 为每只股票生成信号
        for symbol in stock_codes:
            if symbol in stock_data:
                price_data = self._get_price_from_data(symbol, date, stock_data[symbol])
                if price_data:
                    signal = await self.signal_generator.generate_single_signal(
                        symbol, date, price_data
                    )
                    signals.append(signal)

        return signals

    def _get_price_from_data(
        self, symbol: str, date: datetime, df: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """
        从数据中获取价格信息

        Args:
            symbol: 股票代码
            date: 查询日期
            df: 股票数据

        Returns:
            价格数据
        """
        try:
            if date in df.index:
                row = df.loc[date]
                return {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
        except:
            pass

        return None

    def _calculate_total_assets(
        self, date: datetime, stock_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        计算总资产

        Args:
            date: 查询日期
            stock_data: 股票数据

        Returns:
            总资产
        """
        total_assets = self.current_cash

        for symbol, quantity in self.current_positions.items():
            if quantity > 0 and symbol in stock_data:
                price_data = self._get_price_from_data(symbol, date, stock_data[symbol])
                if price_data:
                    total_assets += quantity * price_data["close"]

        return total_assets

    def _calculate_daily_return(self, current_assets: float) -> float:
        """
        计算日收益率

        Args:
            current_assets: 当前总资产

        Returns:
            日收益率
        """
        if len(self.daily_results) == 0:
            return 0.0

        previous_assets = self.daily_results[-1].total_assets
        if previous_assets > 0:
            return (current_assets - previous_assets) / previous_assets

        return 0.0

    def _calculate_backtest_result(
        self, start_date: datetime, end_date: datetime
    ) -> BacktestResult:
        """
        计算回测结果

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果
        """
        if not self.daily_results:
            return self._create_empty_result(start_date, end_date)

        # 基本统计
        final_assets = self.daily_results[-1].total_assets
        total_return = final_assets / self.initial_cash - 1

        # 日收益率序列
        daily_returns = [result.daily_return for result in self.daily_results]

        # 计算年化收益率
        trading_days = len(self.daily_results)
        if trading_days > 0:
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        else:
            annualized_return = 0

        # 计算波动率
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0

        # 计算夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        )

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 计算交易统计
        trade_stats = self._calculate_trade_stats()

        # 构建资产曲线
        equity_curve = [
            {
                "date": result.date,
                "total_assets": result.total_assets,
                "cash": result.cash,
                "positions": result.positions,
                "daily_return": result.daily_return,
            }
            for result in self.daily_results
        ]

        return BacktestResult(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_cash=self.initial_cash,
            final_cash=self.current_cash,
            final_assets=final_assets,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=trade_stats["win_rate"],
            profit_loss_ratio=trade_stats["profit_loss_ratio"],
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            average_trade_return=trade_stats["average_trade_return"],
            volatility=volatility,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            trade_history=self.trade_history,
        )

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.daily_results:
            return 0.0

        assets = [result.total_assets for result in self.daily_results]
        peak = assets[0]
        max_drawdown = 0.0

        for asset in assets:
            if asset > peak:
                peak = asset
            else:
                drawdown = (peak - asset) / peak
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_trade_stats(self) -> Dict[str, Any]:
        """计算交易统计"""
        if not self.trade_history:
            return {
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "average_trade_return": 0.0,
            }

        # 计算每笔交易的收益
        trade_returns = []
        buy_trades = {}

        for trade in self.trade_history:
            symbol = trade["symbol"]
            if trade["action"] == "buy":
                buy_trades[symbol] = trade
            elif trade["action"] == "sell" and symbol in buy_trades:
                buy_trade = buy_trades[symbol]
                buy_price = buy_trade["price"]
                sell_price = trade["price"]
                trade_return = (sell_price - buy_price) / buy_price
                trade_returns.append(trade_return)
                del buy_trades[symbol]

        if not trade_returns:
            return {
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "average_trade_return": 0.0,
            }

        # 统计
        winning_trades = len([r for r in trade_returns if r > 0])
        losing_trades = len([r for r in trade_returns if r < 0])
        total_trades = len(trade_returns)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        average_trade_return = np.mean(trade_returns) if trade_returns else 0

        # 盈亏比
        winning_returns = [r for r in trade_returns if r > 0]
        losing_returns = [abs(r) for r in trade_returns if r < 0]

        if winning_returns and losing_returns:
            profit_loss_ratio = np.mean(winning_returns) / np.mean(losing_returns)
        else:
            profit_loss_ratio = 0.0

        return {
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "average_trade_return": average_trade_return,
        }

    def _create_empty_result(
        self, start_date: datetime, end_date: datetime
    ) -> BacktestResult:
        """创建空结果"""
        return BacktestResult(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_cash=self.initial_cash,
            final_cash=self.initial_cash,
            final_assets=self.initial_cash,
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_trade_return=0.0,
            volatility=0.0,
            daily_returns=[],
            equity_curve=[],
            trade_history=[],
        )


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_backtest():
        """测试回测引擎"""
        # 配置
        config = {
            "initial_cash": 100000.0,
            "max_positions": 10,
            "commission_rate": 0.0003,
            "tdx_path": "C:/F/newtdx",
        }

        # 初始化回测引擎
        engine = BacktestEngine(config)

        # 测试股票池
        stock_codes = ["000001", "000002", "000003", "000004", "000005"]

        # 回测期间
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        print("=" * 80)
        print("🤖 AI智能体回测引擎测试")
        print("=" * 80)
        print(
            f"📅 回测期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"💰 初始资金: {config['initial_cash']:,.2f} 元")
        print(f"📊 股票池: {len(stock_codes)} 只股票")
        print(f"🎯 最大持仓: {config['max_positions']} 只")

        # 运行回测
        result = await engine.run_backtest(stock_codes, start_date, end_date)

        # 输出结果
        print("\n" + "=" * 80)
        print("📊 回测结果")
        print("=" * 80)
        print(f"💰 初始资金: {result.initial_cash:,.2f} 元")
        print(f"💰 最终资产: {result.final_assets:,.2f} 元")
        print(f"📈 总收益率: {result.total_return:.2%}")
        print(f"📈 年化收益率: {result.annualized_return:.2%}")
        print(f"📉 最大回撤: {result.max_drawdown:.2%}")
        print(f"📊 夏普比率: {result.sharpe_ratio:.2f}")
        print(f"📊 波动率: {result.volatility:.2%}")
        print(f"🏆 胜率: {result.win_rate:.2%}")
        print(f"💎 盈亏比: {result.profit_loss_ratio:.2f}")
        print(f"📈 总交易次数: {result.total_trades}")
        print(f"✅ 盈利交易: {result.winning_trades}")
        print(f"❌ 亏损交易: {result.losing_trades}")
        print(f"📊 平均交易收益: {result.average_trade_return:.2%}")

        # 保存结果
        results_file = (
            f"ai_agent_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2, default=str)

        print(f"\n📄 详细结果已保存: {results_file}")
        print("\n🎉 回测测试完成！")

    # 运行测试
    asyncio.run(test_backtest())
