#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI智能体投资策略系统
基于北京大学光华管理学院陈泽丰老师和蒲定磐老师关于 Agentic AI 的前沿研究
实现自主信息收集、分析并做出投资决策的AI系统

核心策略：
1. AI在每个交易日收盘后到下一个交易日开盘前收集信息
2. 信息来源：实时新闻、社交媒体、公司财报等非结构化文本数据
3. AI对每只股票进行综合分析并打分（-5分到+5分）
4. 每个交易日开盘时，买入评分最高的20只股票，下一个交易日开盘时全部卖出
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import requests
from dotenv import load_dotenv

# 添加AI-Trader路径
sys.path.append("C:/F/AI-Trader-main")
load_dotenv("C:/F/AI-Trader-main/.env")

# 导入AI-Trader组件
from tools.general_tools import get_config_value, write_config_value
from agent_tools.tool_jina_search import get_information
from agent_tools.tool_get_price_local import get_price_local
from agent_tools.tool_trade import buy, sell

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ai_agent_strategy.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class StockAnalysis:
    """股票分析结果"""

    symbol: str
    score: float  # -5 到 +5
    reasoning: str  # 分析理由
    confidence: float  # 置信度 0-1
    news_sentiment: float  # 新闻情绪 -1到1
    technical_signal: float  # 技术信号 -1到1
    fundamental_score: float  # 基本面评分 -1到1
    timestamp: datetime


@dataclass
class TradingDecision:
    """交易决策"""

    action: str  # 'buy' or 'sell'
    symbol: str
    quantity: int
    reason: str
    confidence: float
    timestamp: datetime


class AIAgentStrategy:
    """AI智能体投资策略核心类"""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        max_positions: int = 20,
        stock_pool_file: str = "C:/F/stock_pool.txt",
        tdx_data_path: str = "C:/F/newtdx",
    ):
        """
        初始化AI智能体策略

        Args:
            initial_cash: 初始资金
            max_positions: 最大持仓数量
            stock_pool_file: 股票池文件路径
            tdx_data_path: 通达信数据路径
        """
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.stock_pool_file = stock_pool_file
        self.tdx_data_path = tdx_data_path

        # 加载股票池
        self.stock_pool = self._load_stock_pool()

        # 当前持仓
        self.positions = {}
        self.cash = initial_cash

        # 交易历史
        self.trading_history = []

        # AI模型配置
        self.llm_config = {
            "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4"),
            "max_tokens": 2000,
            "temperature": 0.3,
        }

        # 数据获取器
        self.data_reader = TDXDayReader(tdx_data_path)

        logger.info(f"AI智能体策略初始化完成，股票池数量: {len(self.stock_pool)}")

    def _load_stock_pool(self) -> List[str]:
        """加载股票池"""
        try:
            with open(self.stock_pool_file, "r", encoding="utf-8") as f:
                stocks = [line.strip() for line in f if line.strip()]
            logger.info(f"成功加载股票池: {len(stocks)}只股票")
            return stocks
        except Exception as e:
            logger.error(f"加载股票池失败: {e}")
            return []

    async def collect_market_information(
        self, symbol: str, date: datetime
    ) -> Dict[str, Any]:
        """
        收集市场信息

        Args:
            symbol: 股票代码
            date: 查询日期

        Returns:
            市场信息字典
        """
        try:
            # 构建搜索查询
            search_query = f"{symbol} 股票 新闻 财报 市场 {date.strftime('%Y-%m-%d')}"

            # 使用Jina AI搜索信息
            search_results = await get_information(search_query)

            # 获取价格数据
            price_data = await get_price_local(symbol, date.strftime("%Y-%m-%d"))

            # 获取技术指标数据
            technical_data = self._get_technical_indicators(symbol, date)

            return {
                "symbol": symbol,
                "date": date,
                "search_results": search_results,
                "price_data": price_data,
                "technical_data": technical_data,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"收集{symbol}市场信息失败: {e}")
            return {}

    def _get_technical_indicators(
        self, symbol: str, date: datetime
    ) -> Dict[str, float]:
        """
        获取技术指标

        Args:
            symbol: 股票代码
            date: 查询日期

        Returns:
            技术指标字典
        """
        try:
            # 获取历史数据
            end_date = date
            start_date = date - timedelta(days=60)

            df = self.data_reader.read_day_data(symbol, start_date, end_date)

            if df is None or len(df) < 20:
                return {}

            # 计算技术指标
            df = self._calculate_indicators(df)

            # 获取最新数据
            latest = df.iloc[-1]

            return {
                "rsi": latest.get("rsi", 50),
                "macd": latest.get("macd", 0),
                "macd_signal": latest.get("macd_signal", 0),
                "ma5": latest.get("ma5", latest["close"]),
                "ma20": latest.get("ma20", latest["close"]),
                "volume_ratio": latest.get("volume_ratio", 1.0),
                "price_change_pct": latest.get("price_change_pct", 0),
            }

        except Exception as e:
            logger.error(f"获取{symbol}技术指标失败: {e}")
            return {}

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["close"].ewm(span=12).mean()
        exp2 = df["close"].ewm(span=26).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        # 移动平均线
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()

        # 成交量比率
        df["volume_ma"] = df["volume"].rolling(window=5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # 价格变化百分比
        df["price_change_pct"] = df["close"].pct_change() * 100

        return df

    async def analyze_stock(self, symbol: str, date: datetime) -> StockAnalysis:
        """
        分析单只股票

        Args:
            symbol: 股票代码
            date: 分析日期

        Returns:
            股票分析结果
        """
        try:
            # 收集市场信息
            market_info = await self.collect_market_information(symbol, date)

            if not market_info:
                return StockAnalysis(
                    symbol=symbol,
                    score=0.0,
                    reasoning="无法获取市场信息",
                    confidence=0.0,
                    news_sentiment=0.0,
                    technical_signal=0.0,
                    fundamental_score=0.0,
                    timestamp=datetime.now(),
                )

            # 使用AI进行综合分析
            analysis_result = await self._ai_analysis(market_info)

            return StockAnalysis(
                symbol=symbol,
                score=analysis_result.get("score", 0.0),
                reasoning=analysis_result.get("reasoning", ""),
                confidence=analysis_result.get("confidence", 0.0),
                news_sentiment=analysis_result.get("news_sentiment", 0.0),
                technical_signal=analysis_result.get("technical_signal", 0.0),
                fundamental_score=analysis_result.get("fundamental_score", 0.0),
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"分析{symbol}失败: {e}")
            return StockAnalysis(
                symbol=symbol,
                score=0.0,
                reasoning=f"分析失败: {str(e)}",
                confidence=0.0,
                news_sentiment=0.0,
                technical_signal=0.0,
                fundamental_score=0.0,
                timestamp=datetime.now(),
            )

    async def _ai_analysis(self, market_info: Dict[str, Any]) -> Dict[str, float]:
        """
        使用AI进行分析

        Args:
            market_info: 市场信息

        Returns:
            分析结果
        """
        try:
            # 构建分析提示词
            prompt = self._build_analysis_prompt(market_info)

            # 调用AI API
            response = await self._call_ai_api(prompt)

            # 解析AI响应
            result = self._parse_ai_response(response)

            return result

        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return {
                "score": 0.0,
                "confidence": 0.0,
                "news_sentiment": 0.0,
                "technical_signal": 0.0,
                "fundamental_score": 0.0,
                "reasoning": f"AI分析失败: {str(e)}",
            }

    def _build_analysis_prompt(self, market_info: Dict[str, Any]) -> str:
        """构建AI分析提示词"""
        symbol = market_info.get("symbol", "")
        date = market_info.get("date", datetime.now()).strftime("%Y-%m-%d")

        # 提取关键信息
        search_results = market_info.get("search_results", {})
        price_data = market_info.get("price_data", {})
        technical_data = market_info.get("technical_data", {})

        prompt = f"""
你是一位专业的AI股票分析师，请对股票 {symbol} 在 {date} 的投资价值进行综合分析。

## 可用信息：

### 1. 新闻和市场信息
{json.dumps(search_results, ensure_ascii=False, indent=2)[:1000]}

### 2. 价格数据
{json.dumps(price_data, ensure_ascii=False, indent=2)}

### 3. 技术指标
{json.dumps(technical_data, ensure_ascii=False, indent=2)}

## 分析要求：

请基于以上信息，从以下维度进行分析：

1. **新闻情绪分析** (-1到1，-1极度负面，1极度正面)
2. **技术信号分析** (-1到1，-1强烈看跌，1强烈看涨)
3. **基本面评分** (-1到1，-1基本面差，1基本面好)

## 输出格式：

请严格按照以下JSON格式输出分析结果：

```json
{{
    "score": -5.0到5.0的综合评分,
    "news_sentiment": -1.0到1.0的新闻情绪,
    "technical_signal": -1.0到1.0的技术信号,
    "fundamental_score": -1.0到1.0的基本面评分,
    "confidence": 0.0到1.0的置信度,
    "reasoning": "详细的分析理由，包含关键信息点"
}}
```

## 评分标准：
- 综合评分：-5(强烈卖出) 到 +5(强烈买入)
- 置信度：0(不确定) 到 1(非常确定)
- 分析理由：要包含具体的新闻事件、技术指标、基本面因素

请确保输出的是有效的JSON格式。
"""
        return prompt

    async def _call_ai_api(self, prompt: str) -> str:
        """调用AI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.llm_config['api_key']}",
                "Content-Type": "application/json",
            }

            data = {
                "model": self.llm_config["model"],
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位专业的股票分析师，请严格按照JSON格式输出分析结果。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.llm_config["max_tokens"],
                "temperature": self.llm_config["temperature"],
            }

            response = requests.post(
                f"{self.llm_config['api_base']}/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"AI API调用失败: {response.status_code}, {response.text}")
                return "{}"

        except Exception as e:
            logger.error(f"调用AI API异常: {e}")
            return "{}"

    def _parse_ai_response(self, response: str) -> Dict[str, float]:
        """解析AI响应"""
        try:
            # 提取JSON部分
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)

                # 验证和标准化数据
                return {
                    "score": float(result.get("score", 0.0)),
                    "news_sentiment": float(result.get("news_sentiment", 0.0)),
                    "technical_signal": float(result.get("technical_signal", 0.0)),
                    "fundamental_score": float(result.get("fundamental_score", 0.0)),
                    "confidence": float(result.get("confidence", 0.0)),
                    "reasoning": result.get("reasoning", ""),
                }
            else:
                logger.error("AI响应中未找到有效的JSON")
                return {
                    "score": 0.0,
                    "news_sentiment": 0.0,
                    "technical_signal": 0.0,
                    "fundamental_score": 0.0,
                    "confidence": 0.0,
                    "reasoning": "AI响应格式错误",
                }

        except Exception as e:
            logger.error(f"解析AI响应失败: {e}")
            return {
                "score": 0.0,
                "news_sentiment": 0.0,
                "technical_signal": 0.0,
                "fundamental_score": 0.0,
                "confidence": 0.0,
                "reasoning": f"解析失败: {str(e)}",
            }

    async def analyze_all_stocks(self, date: datetime) -> List[StockAnalysis]:
        """
        分析所有股票

        Args:
            date: 分析日期

        Returns:
            所有股票的分析结果
        """
        logger.info(
            f"开始分析{len(self.stock_pool)}只股票，日期: {date.strftime('%Y-%m-%d')}"
        )

        results = []

        # 并发分析（限制并发数避免API限制）
        semaphore = asyncio.Semaphore(5)

        async def analyze_with_semaphore(symbol):
            async with semaphore:
                return await self.analyze_stock(symbol, date)

        tasks = [analyze_with_semaphore(symbol) for symbol in self.stock_pool]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤异常结果
        valid_results = []
        for result in results:
            if isinstance(result, StockAnalysis):
                valid_results.append(result)
            else:
                logger.error(f"分析结果异常: {result}")

        logger.info(f"完成分析，有效结果: {len(valid_results)}只股票")
        return valid_results

    def select_top_stocks(self, analyses: List[StockAnalysis]) -> List[StockAnalysis]:
        """
        选择评分最高的股票

        Args:
            analyses: 股票分析结果列表

        Returns:
            评分最高的股票列表
        """
        # 按评分排序
        sorted_analyses = sorted(analyses, key=lambda x: x.score, reverse=True)

        # 过滤低置信度和负评分的股票
        filtered_analyses = [
            analysis
            for analysis in sorted_analyses
            if analysis.confidence > 0.5 and analysis.score > 0
        ]

        # 返回前N只
        top_stocks = filtered_analyses[: self.max_positions]

        logger.info(
            f"选择前{len(top_stocks)}只股票，评分范围: {top_stocks[0].score:.2f} 到 {top_stocks[-1].score:.2f}"
        )

        return top_stocks

    async def make_trading_decisions(self, date: datetime) -> List[TradingDecision]:
        """
        做出交易决策

        Args:
            date: 交易日期

        Returns:
            交易决策列表
        """
        logger.info(f"开始为{date.strftime('%Y-%m-%d')}做出交易决策")

        # 分析所有股票
        analyses = await self.analyze_all_stocks(date)

        # 选择最佳股票
        top_stocks = self.select_top_stocks(analyses)

        decisions = []

        # 卖出决策：清空所有持仓
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                decisions.append(
                    TradingDecision(
                        action="sell",
                        symbol=symbol,
                        quantity=quantity,
                        reason="每日清仓策略",
                        confidence=1.0,
                        timestamp=date,
                    )
                )

        # 买入决策：买入评分最高的股票
        available_cash = self.cash
        for i, analysis in enumerate(top_stocks):
            if available_cash <= 1000:  # 保留1000元作为缓冲
                break

            # 等权重分配资金
            position_size = available_cash / (len(top_stocks) - i)

            # 获取当前价格
            price_data = await get_price_local(
                analysis.symbol, date.strftime("%Y-%m-%d")
            )
            if price_data and "close" in price_data:
                current_price = price_data["close"]
                quantity = int(position_size / current_price / 100) * 100  # 按手买入

                if quantity > 0:
                    decisions.append(
                        TradingDecision(
                            action="buy",
                            symbol=analysis.symbol,
                            quantity=quantity,
                            reason=f"AI评分: {analysis.score:.2f}, 理由: {analysis.reasoning}",
                            confidence=analysis.confidence,
                            timestamp=date,
                        )
                    )

                    available_cash -= quantity * current_price

        logger.info(f"生成{len(decisions)}个交易决策")
        return decisions

    async def execute_trading_decisions(self, decisions: List[TradingDecision]) -> bool:
        """
        执行交易决策

        Args:
            decisions: 交易决策列表

        Returns:
            执行是否成功
        """
        logger.info(f"开始执行{len(decisions)}个交易决策")

        success_count = 0

        for decision in decisions:
            try:
                if decision.action == "buy":
                    # 执行买入
                    result = await buy(decision.symbol, decision.quantity)
                    if result:
                        self.positions[decision.symbol] = (
                            self.positions.get(decision.symbol, 0) + decision.quantity
                        )
                        self.cash -= decision.quantity * result.get("price", 0)
                        success_count += 1
                        logger.info(
                            f"买入成功: {decision.symbol} {decision.quantity}股"
                        )
                    else:
                        logger.error(f"买入失败: {decision.symbol}")

                elif decision.action == "sell":
                    # 执行卖出
                    result = await sell(decision.symbol, decision.quantity)
                    if result:
                        self.positions[decision.symbol] = (
                            self.positions.get(decision.symbol, 0) - decision.quantity
                        )
                        self.cash += decision.quantity * result.get("price", 0)
                        success_count += 1
                        logger.info(
                            f"卖出成功: {decision.symbol} {decision.quantity}股"
                        )
                    else:
                        logger.error(f"卖出失败: {decision.symbol}")

                # 记录交易历史
                self.trading_history.append(
                    {
                        "decision": asdict(decision),
                        "result": result,
                        "timestamp": datetime.now(),
                    }
                )

            except Exception as e:
                logger.error(f"执行交易决策失败: {decision.symbol}, {e}")

        logger.info(f"交易执行完成，成功: {success_count}/{len(decisions)}")
        return success_count == len(decisions)

    async def run_daily_strategy(self, date: datetime) -> Dict[str, Any]:
        """
        运行每日策略

        Args:
            date: 交易日期

        Returns:
            策略执行结果
        """
        logger.info(f"开始执行{date.strftime('%Y-%m-%d')}的AI智能体策略")

        try:
            # 1. 做出交易决策
            decisions = await self.make_trading_decisions(date)

            # 2. 执行交易决策
            execution_success = await self.execute_trading_decisions(decisions)

            # 3. 计算当前资产
            total_assets = self.cash
            for symbol, quantity in self.positions.items():
                if quantity > 0:
                    price_data = await get_price_local(
                        symbol, date.strftime("%Y-%m-%d")
                    )
                    if price_data and "close" in price_data:
                        total_assets += quantity * price_data["close"]

            # 4. 返回执行结果
            result = {
                "date": date.strftime("%Y-%m-%d"),
                "decisions_count": len(decisions),
                "execution_success": execution_success,
                "cash": self.cash,
                "positions": dict(self.positions),
                "total_assets": total_assets,
                "daily_return": (total_assets / self.initial_cash - 1) * 100,
                "decisions": [asdict(d) for d in decisions],
            }

            logger.info(
                f"策略执行完成，总资产: {total_assets:.2f}, 日收益率: {result['daily_return']:.2f}%"
            )

            return result

        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            return {
                "date": date.strftime("%Y-%m-%d"),
                "error": str(e),
                "success": False,
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取策略表现摘要"""
        if not self.trading_history:
            return {"message": "暂无交易历史"}

        # 计算基本统计
        total_trades = len(self.trading_history)
        buy_trades = len(
            [t for t in self.trading_history if t["decision"]["action"] == "buy"]
        )
        sell_trades = len(
            [t for t in self.trading_history if t["decision"]["action"] == "sell"]
        )

        # 计算当前资产价值
        current_assets = self.cash

        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "current_cash": self.cash,
            "current_positions": dict(self.positions),
            "current_assets": current_assets,
            "initial_cash": self.initial_cash,
            "total_return": (current_assets / self.initial_cash - 1) * 100,
        }


class TDXDayReader:
    """通达信日线数据读取器（从现有代码复制）"""

    def __init__(self, tdx_path="C:/F/newtdx"):
        self.tdx_path = Path(tdx_path)
        self.vipdoc_path = self.tdx_path / "vipdoc"

    def get_day_file_path(self, stock_code):
        """获取股票日线文件路径"""
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

    def read_day_data(self, stock_code, start_date=None, end_date=None):
        """读取日线数据，返回DataFrame"""
        import struct

        file_path = self.get_day_file_path(stock_code)

        if not file_path or not file_path.exists():
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
                    except:
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
            return None

        except Exception as e:
            return None


async def main():
    """主函数"""
    print("=" * 80)
    print("🤖 AI智能体投资策略系统")
    print("基于北京大学光华管理学院前沿研究")
    print("=" * 80)

    # 初始化策略
    strategy = AIAgentStrategy(
        initial_cash=100000.0,
        max_positions=20,
        stock_pool_file="C:/F/stock_pool.txt",
        tdx_data_path="C:/F/newtdx",
    )

    # 设置测试日期范围
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    print(
        f"📅 回测期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"💰 初始资金: {strategy.initial_cash:,.2f} 元")
    print(f"📊 股票池数量: {len(strategy.stock_pool)} 只")
    print(f"🎯 最大持仓: {strategy.max_positions} 只")

    # 运行回测
    results = []
    current_date = start_date

    while current_date <= end_date:
        # 跳过周末
        if current_date.weekday() < 5:  # 0-4 表示周一到周五
            print(f"\n📈 执行 {current_date.strftime('%Y-%m-%d')} 策略...")
            result = await strategy.run_daily_strategy(current_date)
            results.append(result)

            if "error" not in result:
                print(f"   总资产: {result['total_assets']:,.2f} 元")
                print(f"   收益率: {result['daily_return']:.2f}%")
            else:
                print(f"   执行失败: {result['error']}")

        current_date += timedelta(days=1)

    # 输出最终结果
    print("\n" + "=" * 80)
    print("📊 回测结果汇总")
    print("=" * 80)

    if results:
        final_result = results[-1]
        performance = strategy.get_performance_summary()

        print(
            f"📅 回测期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"💰 初始资金: {strategy.initial_cash:,.2f} 元")
        print(f"💰 最终资产: {performance['current_assets']:,.2f} 元")
        print(f"📈 总收益率: {performance['total_return']:.2f}%")
        print(f"📊 总交易次数: {performance['total_trades']}")
        print(f"📈 买入次数: {performance['buy_trades']}")
        print(f"📉 卖出次数: {performance['sell_trades']}")

        # 保存详细结果
        results_file = (
            f"ai_agent_strategy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "strategy_config": {
                        "initial_cash": strategy.initial_cash,
                        "max_positions": strategy.max_positions,
                        "stock_pool_size": len(strategy.stock_pool),
                    },
                    "backtest_period": {
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                    },
                    "daily_results": results,
                    "final_performance": performance,
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        print(f"📄 详细结果已保存: {results_file}")

    print("\n🎉 AI智能体策略回测完成！")


if __name__ == "__main__":
    asyncio.run(main())
