#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI智能体信息收集与处理模块
简化版本，专注于核心功能实现
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketInfo:
    """市场信息数据结构"""

    symbol: str
    date: datetime
    price_data: Dict[str, float]
    news_sentiment: float
    technical_indicators: Dict[str, float]
    volume_data: Dict[str, float]
    market_context: str


class InformationCollector:
    """信息收集器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化信息收集器

        Args:
            config: 配置信息
        """
        self.config = config or {}
        self.session = requests.Session()

        # API配置
        self.news_api_key = self.config.get("news_api_key", "")
        self.alpha_vantage_key = self.config.get("alpha_vantage_key", "")

        logger.info("信息收集器初始化完成")

    def collect_stock_news(self, symbol: str, date: datetime) -> List[Dict[str, Any]]:
        """
        收集股票新闻

        Args:
            symbol: 股票代码
            date: 查询日期

        Returns:
            新闻列表
        """
        try:
            # 模拟新闻数据（实际应用中应该调用真实API）
            mock_news = [
                {
                    "title": f"{symbol} 公司发布利好消息",
                    "content": "公司业绩超预期，未来前景看好",
                    "sentiment": 0.8,
                    "date": date.strftime("%Y-%m-%d"),
                    "source": "财经新闻",
                },
                {
                    "title": f"{symbol} 行业分析报告",
                    "content": "行业整体向好，公司具备竞争优势",
                    "sentiment": 0.6,
                    "date": date.strftime("%Y-%m-%d"),
                    "source": "研究机构",
                },
            ]

            return mock_news

        except Exception as e:
            logger.error(f"收集{symbol}新闻失败: {e}")
            return []

    def collect_financial_data(self, symbol: str, date: datetime) -> Dict[str, float]:
        """
        收集财务数据

        Args:
            symbol: 股票代码
            date: 查询日期

        Returns:
            财务数据
        """
        try:
            # 模拟财务数据（实际应用中应该调用真实API）
            mock_financial = {
                "revenue_growth": np.random.normal(0.15, 0.05),  # 营收增长率
                "profit_margin": np.random.normal(0.12, 0.03),  # 利润率
                "debt_ratio": np.random.normal(0.4, 0.1),  # 负债率
                "roe": np.random.normal(0.15, 0.04),  # ROE
                "pe_ratio": np.random.normal(20, 5),  # PE比率
                "market_cap": np.random.normal(1000000000, 200000000),  # 市值
            }

            return mock_financial

        except Exception as e:
            logger.error(f"收集{symbol}财务数据失败: {e}")
            return {}

    def collect_market_sentiment(self, symbol: str, date: datetime) -> float:
        """
        收集市场情绪

        Args:
            symbol: 股票代码
            date: 查询日期

        Returns:
            市场情绪分数 (-1 到 1)
        """
        try:
            # 模拟市场情绪数据
            sentiment = np.random.normal(0.2, 0.3)
            return max(-1, min(1, sentiment))

        except Exception as e:
            logger.error(f"收集{symbol}市场情绪失败: {e}")
            return 0.0

    def collect_technical_indicators(
        self, symbol: str, date: datetime, price_data: Dict[str, float]
    ) -> Dict[str, float]:
        """
        收集技术指标

        Args:
            symbol: 股票代码
            date: 查询日期
            price_data: 价格数据

        Returns:
            技术指标
        """
        try:
            # 基于价格数据计算技术指标
            current_price = price_data.get("close", 100)

            # 模拟技术指标
            technical_indicators = {
                "rsi": np.random.normal(50, 15),
                "macd": np.random.normal(0, 2),
                "macd_signal": np.random.normal(0, 1.5),
                "ma5": current_price * np.random.normal(1.0, 0.02),
                "ma20": current_price * np.random.normal(1.0, 0.05),
                "volume_ratio": np.random.normal(1.2, 0.3),
                "price_momentum": np.random.normal(0.01, 0.02),
                "volatility": np.random.normal(0.02, 0.01),
            }

            return technical_indicators

        except Exception as e:
            logger.error(f"收集{symbol}技术指标失败: {e}")
            return {}

    async def collect_all_information(
        self, symbol: str, date: datetime, price_data: Dict[str, float]
    ) -> MarketInfo:
        """
        收集所有信息

        Args:
            symbol: 股票代码
            date: 查询日期
            price_data: 价格数据

        Returns:
            市场信息
        """
        try:
            # 并行收集各类信息
            news = self.collect_stock_news(symbol, date)
            financial = self.collect_financial_data(symbol, date)
            sentiment = self.collect_market_sentiment(symbol, date)
            technical = self.collect_technical_indicators(symbol, date, price_data)

            # 计算新闻情绪
            news_sentiment = np.mean([n["sentiment"] for n in news]) if news else 0.0

            # 构建市场上下文
            market_context = self._build_market_context(news, financial, sentiment)

            return MarketInfo(
                symbol=symbol,
                date=date,
                price_data=price_data,
                news_sentiment=news_sentiment,
                technical_indicators=technical,
                volume_data={"volume": price_data.get("volume", 0)},
                market_context=market_context,
            )

        except Exception as e:
            logger.error(f"收集{symbol}所有信息失败: {e}")
            return MarketInfo(
                symbol=symbol,
                date=date,
                price_data=price_data,
                news_sentiment=0.0,
                technical_indicators={},
                volume_data={},
                market_context=f"信息收集失败: {str(e)}",
            )

    def _build_market_context(
        self, news: List[Dict], financial: Dict, sentiment: float
    ) -> str:
        """构建市场上下文描述"""
        context_parts = []

        # 新闻摘要
        if news:
            context_parts.append(
                f"相关新闻{len(news)}条，主要情绪: {'正面' if sentiment > 0 else '负面'}"
            )

        # 财务摘要
        if financial:
            revenue_growth = financial.get("revenue_growth", 0)
            profit_margin = financial.get("profit_margin", 0)
            context_parts.append(
                f"营收增长: {revenue_growth:.1%}, 利润率: {profit_margin:.1%}"
            )

        # 市场情绪
        context_parts.append(f"市场情绪: {sentiment:.2f}")

        return "; ".join(context_parts)


class InformationProcessor:
    """信息处理器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化信息处理器

        Args:
            config: 配置信息
        """
        self.config = config or {}
        logger.info("信息处理器初始化完成")

    def process_news_sentiment(self, news: List[Dict[str, Any]]) -> float:
        """
        处理新闻情绪

        Args:
            news: 新闻列表

        Returns:
            情绪分数 (-1 到 1)
        """
        if not news:
            return 0.0

        # 计算加权情绪分数
        total_weight = 0
        weighted_sentiment = 0

        for item in news:
            sentiment = item.get("sentiment", 0)
            weight = 1.0  # 可以根据新闻来源、时间等调整权重

            weighted_sentiment += sentiment * weight
            total_weight += weight

        if total_weight > 0:
            return max(-1, min(1, weighted_sentiment / total_weight))

        return 0.0

    def process_technical_signals(
        self, technical_indicators: Dict[str, float]
    ) -> float:
        """
        处理技术信号

        Args:
            technical_indicators: 技术指标

        Returns:
            技术信号分数 (-1 到 1)
        """
        if not technical_indicators:
            return 0.0

        signals = []

        # RSI信号
        rsi = technical_indicators.get("rsi", 50)
        if rsi < 30:
            signals.append(0.8)  # 超卖，买入信号
        elif rsi > 70:
            signals.append(-0.8)  # 超买，卖出信号
        else:
            signals.append(0.0)  # 中性

        # MACD信号
        macd = technical_indicators.get("macd", 0)
        macd_signal = technical_indicators.get("macd_signal", 0)
        if macd > macd_signal:
            signals.append(0.6)  # 金叉，买入信号
        else:
            signals.append(-0.6)  # 死叉，卖出信号

        # 移动平均线信号
        ma5 = technical_indicators.get("ma5", 0)
        ma20 = technical_indicators.get("ma20", 0)
        if ma5 > ma20:
            signals.append(0.4)  # 短期均线上穿长期均线
        else:
            signals.append(-0.4)  # 短期均线下穿长期均线

        # 成交量信号
        volume_ratio = technical_indicators.get("volume_ratio", 1)
        if volume_ratio > 1.5:
            signals.append(0.3)  # 放量
        elif volume_ratio < 0.7:
            signals.append(-0.3)  # 缩量

        # 计算综合技术信号
        if signals:
            return max(-1, min(1, np.mean(signals)))

        return 0.0

    def process_fundamental_score(self, financial_data: Dict[str, float]) -> float:
        """
        处理基本面评分

        Args:
            financial_data: 财务数据

        Returns:
            基本面评分 (-1 到 1)
        """
        if not financial_data:
            return 0.0

        scores = []

        # 营收增长评分
        revenue_growth = financial_data.get("revenue_growth", 0)
        if revenue_growth > 0.2:
            scores.append(0.8)
        elif revenue_growth > 0.1:
            scores.append(0.5)
        elif revenue_growth > 0:
            scores.append(0.2)
        else:
            scores.append(-0.5)

        # 利润率评分
        profit_margin = financial_data.get("profit_margin", 0)
        if profit_margin > 0.15:
            scores.append(0.6)
        elif profit_margin > 0.1:
            scores.append(0.3)
        elif profit_margin > 0.05:
            scores.append(0.1)
        else:
            scores.append(-0.3)

        # ROE评分
        roe = financial_data.get("roe", 0)
        if roe > 0.15:
            scores.append(0.5)
        elif roe > 0.1:
            scores.append(0.3)
        elif roe > 0.05:
            scores.append(0.1)
        else:
            scores.append(-0.2)

        # 负债率评分
        debt_ratio = financial_data.get("debt_ratio", 0.5)
        if debt_ratio < 0.3:
            scores.append(0.3)
        elif debt_ratio < 0.5:
            scores.append(0.1)
        elif debt_ratio < 0.7:
            scores.append(-0.1)
        else:
            scores.append(-0.4)

        # PE比率评分
        pe_ratio = financial_data.get("pe_ratio", 20)
        if 10 < pe_ratio < 25:
            scores.append(0.3)
        elif 5 < pe_ratio < 35:
            scores.append(0.1)
        else:
            scores.append(-0.2)

        if scores:
            return max(-1, min(1, np.mean(scores)))

        return 0.0

    def calculate_comprehensive_score(
        self, market_info: MarketInfo
    ) -> Dict[str, float]:
        """
        计算综合评分

        Args:
            market_info: 市场信息

        Returns:
            综合评分结果
        """
        try:
            # 处理各类信息
            news_sentiment = market_info.news_sentiment
            technical_signal = self.process_technical_signals(
                market_info.technical_indicators
            )

            # 模拟基本面数据（实际应该从market_info中获取）
            mock_financial = {
                "revenue_growth": np.random.normal(0.15, 0.05),
                "profit_margin": np.random.normal(0.12, 0.03),
                "roe": np.random.normal(0.15, 0.04),
                "debt_ratio": np.random.normal(0.4, 0.1),
                "pe_ratio": np.random.normal(20, 5),
            }
            fundamental_score = self.process_fundamental_score(mock_financial)

            # 计算综合评分（加权平均）
            weights = {
                "news_sentiment": 0.3,
                "technical_signal": 0.4,
                "fundamental_score": 0.3,
            }

            comprehensive_score = (
                news_sentiment * weights["news_sentiment"]
                + technical_signal * weights["technical_signal"]
                + fundamental_score * weights["fundamental_score"]
            )

            # 转换为-5到+5的评分范围
            final_score = comprehensive_score * 5

            # 计算置信度
            confidence = self._calculate_confidence(
                news_sentiment, technical_signal, fundamental_score
            )

            return {
                "score": final_score,
                "news_sentiment": news_sentiment,
                "technical_signal": technical_signal,
                "fundamental_score": fundamental_score,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"计算{market_info.symbol}综合评分失败: {e}")
            return {
                "score": 0.0,
                "news_sentiment": 0.0,
                "technical_signal": 0.0,
                "fundamental_score": 0.0,
                "confidence": 0.0,
            }

    def _calculate_confidence(self, *scores) -> float:
        """
        计算置信度

        Args:
            scores: 各项评分

        Returns:
            置信度 (0 到 1)
        """
        # 基于评分的一致性计算置信度
        if len(scores) < 2:
            return 0.5

        # 计算评分的标准差
        std_dev = np.std(scores)

        # 标准差越小，置信度越高
        confidence = max(0.1, min(1.0, 1.0 - std_dev / 2.0))

        return confidence


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_information_collector():
        """测试信息收集器"""
        collector = InformationCollector()
        processor = InformationProcessor()

        # 测试数据
        symbol = "000001"
        date = datetime(2024, 1, 15)
        price_data = {
            "open": 10.5,
            "high": 11.0,
            "low": 10.2,
            "close": 10.8,
            "volume": 1000000,
        }

        # 收集信息
        market_info = await collector.collect_all_information(symbol, date, price_data)

        print(f"股票: {market_info.symbol}")
        print(f"日期: {market_info.date}")
        print(f"新闻情绪: {market_info.news_sentiment:.2f}")
        print(f"技术指标: {market_info.technical_indicators}")
        print(f"市场上下文: {market_info.market_context}")

        # 处理信息
        scores = processor.calculate_comprehensive_score(market_info)

        print(f"\n综合评分:")
        print(f"  总评分: {scores['score']:.2f}")
        print(f"  新闻情绪: {scores['news_sentiment']:.2f}")
        print(f"  技术信号: {scores['technical_signal']:.2f}")
        print(f"  基本面评分: {scores['fundamental_score']:.2f}")
        print(f"  置信度: {scores['confidence']:.2f}")

    # 运行测试
    asyncio.run(test_information_collector())
