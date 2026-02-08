#!/usr/bin/python
"""
AI智能体策略数据源管理器
支持TQSDK、本地通达信数据源，以及GLM-4.7-flash免费模型
"""

import os
import sys
import struct
import numpy as np
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# 尝试导入TQSDK
try:
    from tqsdk import TqApi, TqAuth

    TQSDK_AVAILABLE = True
    print("✓ TQSDK可用")
except ImportError:
    TQSDK_AVAILABLE = False
    print("⚠ TQSDK不可用，将使用本地数据源")


class DataSourceManager:
    """数据源管理器"""

    def __init__(self, config_file: str = "data_source_config.json"):
        self.config_file = config_file
        self.config = self.load_config()

        # 数据源客户端
        self.tq_api = None
        self.tdx_data_cache = {}

        # 初始化数据源
        self.initialize_data_sources()

        print("数据源管理器初始化完成")

    def load_config(self) -> Dict:
        """加载数据源配置"""
        default_config = {
            "data_sources": {
                "tqsdk": {
                    "enabled": TQSDK_AVAILABLE,
                    "account": "magicday",
                    "password": "334455",
                    "web_gui": False,
                    "priority": 1,
                },
                "tdx_local": {
                    "enabled": True,
                    "data_path": "/海王星/vipdoc",
                    "priority": 2,
                    "cache_enabled": True,
                    "cache_ttl": 300,
                },
                "tdx_remote": {"enabled": False, "api_endpoint": "", "priority": 3},
            },
            "stock_pools": {
                "main_pool": "C:/F/stock_pool_2509.txt",
                "test_pool": "C:/F/test_stock_pool.txt",
                "active_pool": "main_pool",
            },
            "data_format": {
                "default_period": "1d",
                "data_length": 100,
                "update_frequency": 60,
            },
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                config = {**default_config, **loaded_config}
                print(f"✓ 已加载数据源配置: {self.config_file}")
            else:
                config = default_config
                self.save_config(config)
                print(f"✓ 创建默认数据源配置: {self.config_file}")
        except Exception as e:
            print(f"⚠ 数据源配置加载失败，使用默认配置: {e}")
            config = default_config

        return config

    def save_config(self, config: Dict = None):
        """保存数据源配置"""
        if config is None:
            config = self.config

        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✓ 数据源配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"⚠ 数据源配置保存失败: {e}")

    def initialize_data_sources(self):
        """初始化数据源"""
        # 初始化TQSDK
        if self.config["data_sources"]["tqsdk"]["enabled"] and TQSDK_AVAILABLE:
            try:
                tq_config = self.config["data_sources"]["tqsdk"]
                self.tq_api = TqApi(
                    web_gui=tq_config["web_gui"],
                    auth=TqAuth(tq_config["account"], tq_config["password"]),
                )
                print("✓ TQSDK初始化成功")
            except Exception as e:
                print(f"⚠ TQSDK初始化失败: {e}")
                self.config["data_sources"]["tqsdk"]["enabled"] = False

        # 检查本地通达信数据
        if self.config["data_sources"]["tdx_local"]["enabled"]:
            tdx_path = self.config["data_sources"]["tdx_local"]["data_path"]
            if os.path.exists(tdx_path):
                print(f"✓ 本地通达信数据路径可用: {tdx_path}")
            else:
                print(f"⚠ 本地通达信数据路径不存在: {tdx_path}")
                self.config["data_sources"]["tdx_local"]["enabled"] = False

    def get_stock_data(
        self, stock_code: str, period: str = "1d", data_length: int = 100
    ) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        # 按优先级尝试数据源
        data_sources = sorted(
            self.config["data_sources"].items(), key=lambda x: x[1]["priority"]
        )

        for source_name, source_config in data_sources:
            if not source_config["enabled"]:
                continue

            try:
                if source_name == "tqsdk":
                    data = self._get_tqsdk_data(stock_code, period, data_length)
                elif source_name == "tdx_local":
                    data = self._get_tdx_local_data(stock_code, period, data_length)
                elif source_name == "tdx_remote":
                    data = self._get_tdx_remote_data(stock_code, period, data_length)
                else:
                    continue

                if data is not None and not data.empty:
                    print(f"✓ 从{source_name}获取{stock_code}数据成功")
                    return data

            except Exception as e:
                print(f"⚠ 从{source_name}获取{stock_code}数据失败: {e}")
                continue

        print(f"❌ 所有数据源都无法获取{stock_code}数据")
        return None

    def _get_tqsdk_data(
        self, stock_code: str, period: str, data_length: int
    ) -> Optional[pd.DataFrame]:
        """从TQSDK获取数据"""
        if not self.tq_api:
            return None

        try:
            # 转换股票代码格式
            tq_symbol = self._convert_to_tq_symbol(stock_code)

            # 获取K线数据
            klines = self.tq_api.get_kline_serial(tq_symbol, data_length)

            if klines is None or klines.empty:
                return None

            # 转换数据格式
            data = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(klines["datetime"] / 1e9, unit="s"),
                    "open": klines["open"],
                    "high": klines["high"],
                    "low": klines["low"],
                    "close": klines["close"],
                    "volume": klines["volume"],
                    "amount": klines.get("amount", 0),
                }
            )

            return data.sort_values("datetime").tail(data_length)

        except Exception as e:
            print(f"TQSDK获取数据失败: {e}")
            return None

    def _get_tdx_local_data(
        self, stock_code: str, period: str, data_length: int
    ) -> Optional[pd.DataFrame]:
        """从本地通达信获取数据"""
        try:
            # 转换股票代码格式
            tdx_symbol = self._convert_to_tdx_symbol(stock_code)

            # 构建文件路径
            tdx_path = self.config["data_sources"]["tdx_local"]["data_path"]
            file_path = self._build_tdx_file_path(tdx_path, tdx_symbol, period)

            if not os.path.exists(file_path):
                return None

            # 检查缓存
            cache_key = f"{stock_code}_{period}_{data_length}"
            if self._is_cache_valid(cache_key):
                return self.tdx_data_cache[cache_key]["data"]

            # 读取通达信文件
            data = self._read_tdx_file(file_path, stock_code)

            if data is not None and not data.empty:
                # 缓存数据
                if self.config["data_sources"]["tdx_local"]["cache_enabled"]:
                    self.tdx_data_cache[cache_key] = {
                        "data": data,
                        "timestamp": time.time(),
                    }

                return data.tail(data_length)

            return None

        except Exception as e:
            print(f"本地通达信获取数据失败: {e}")
            return None

    def _get_tdx_remote_data(
        self, stock_code: str, period: str, data_length: int
    ) -> Optional[pd.DataFrame]:
        """从远程通达信获取数据"""
        # 这里可以实现远程通达信API调用
        # 暂时返回None
        return None

    def _convert_to_tq_symbol(self, stock_code: str) -> str:
        """转换到TQSDK格式"""
        # 移除后缀
        code = stock_code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")

        # 根据代码判断交易所
        if code.startswith("00") or code.startswith("30"):
            return f"SSE.{code}"  # 上交所
        elif code.startswith("60") or code.startswith("68"):
            return f"SZSE.{code}"  # 深交所
        elif code.startswith("8") or code.startswith("9"):
            return f"CFFEX.{code}"  # 中金所
        else:
            return f"SSE.{code}"  # 默认上交所

    def _convert_to_tdx_symbol(self, stock_code: str) -> str:
        """转换到通达信格式"""
        # 移除后缀
        code = stock_code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")

        # 根据代码判断市场
        if code.startswith("00") or code.startswith("30"):
            return code  # 深圳市场
        elif code.startswith("60") or code.startswith("68"):
            return code  # 上海市场
        elif code.startswith("8") or code.startswith("9"):
            return code  # 北京市场
        else:
            return code  # 默认

    def _build_tdx_file_path(self, tdx_path: str, symbol: str, period: str) -> str:
        """构建通达信文件路径"""
        if period == "1d":
            # 日线文件
            if symbol.startswith("00") or symbol.startswith("30"):
                market = "sz"
            elif symbol.startswith("60") or symbol.startswith("68"):
                market = "sh"
            else:
                market = "sh"  # 默认

            return f"{tdx_path}/{market}/lday/{symbol}.day"
        else:
            # 其他周期文件
            return f"{tdx_path}/{symbol}.{period}"

    def _read_tdx_file(self, file_path: str, stock_code: str) -> Optional[pd.DataFrame]:
        """读取通达信文件"""
        try:
            with open(file_path, "rb") as f:
                buffer = f.read()

            size = len(buffer)
            row_size = 32  # 通达信日线数据每32字节
            data_set = []

            for i in range(0, size, row_size):
                if i + row_size > size:
                    break

                # 解析一行数据
                row_data = struct.unpack("IIIIIfII", buffer[i : i + row_size])

                # 转换数据格式
                date_int = row_data[0]
                open_price = row_data[1] / 100.0
                high_price = row_data[2] / 100.0
                low_price = row_data[3] / 100.0
                close_price = row_data[4] / 100.0
                amount = row_data[5] / 10.0
                volume = row_data[6]

                # 转换日期
                trade_date = datetime.strptime(str(date_int), "%Y%m%d")

                data_set.append(
                    [
                        stock_code,
                        trade_date,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume,
                        amount,
                    ]
                )

            # 创建DataFrame
            columns = [
                "code",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
            ]
            data = pd.DataFrame(data_set, columns=columns)

            return data.sort_values("datetime")

        except Exception as e:
            print(f"读取通达信文件失败: {e}")
            return None

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.tdx_data_cache:
            return False

        cache_time = self.tdx_data_cache[cache_key]["timestamp"]
        cache_ttl = self.config["data_sources"]["tdx_local"]["cache_ttl"]

        return (time.time() - cache_time) < cache_ttl

    def get_stock_pool(self, pool_name: str = None) -> List[str]:
        """获取股票池"""
        if pool_name is None:
            pool_name = self.config["stock_pools"]["active_pool"]

        pool_file = self.config["stock_pools"].get(pool_name)
        if not pool_file or not os.path.exists(pool_file):
            print(f"⚠ 股票池文件不存在: {pool_file}")
            return []

        try:
            stock_pool = []
            with open(pool_file, "r", encoding="utf-8") as f:
                for line in f:
                    code = line.strip()
                    if code:
                        stock_pool.append(code)

            print(f"✓ 从{pool_file}加载股票池，共{len(stock_pool)}只股票")
            return stock_pool

        except Exception as e:
            print(f"⚠ 加载股票池失败: {e}")
            return []

    def batch_get_stock_data(
        self, stock_codes: List[str], period: str = "1d", data_length: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """批量获取股票数据"""
        results = {}

        print(f"开始批量获取{len(stock_codes)}只股票数据...")

        for i, stock_code in enumerate(stock_codes):
            print(f"获取进度: {i + 1}/{len(stock_codes)} - {stock_code}")

            data = self.get_stock_data(stock_code, period, data_length)
            if data is not None:
                results[stock_code] = data

            # 避免请求过于频繁
            time.sleep(0.1)

        print(f"✓ 批量获取完成，成功获取{len(results)}只股票数据")
        return results

    def close(self):
        """关闭数据源连接"""
        if self.tq_api:
            try:
                self.tq_api.close()
                print("✓ TQSDK连接已关闭")
            except:
                pass


class GLMFlashClient:
    """GLM-4.7-flash免费模型客户端"""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or "4db0d99270664530b2ec62e4862f0f8e.STEfVsL3x4M4m7Jn"
        self.base_url = (
            base_url or "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        print("GLM-4.7-flash客户端初始化完成")

    def analyze_stock(self, stock_data: Dict, factors: Dict) -> Dict:
        """使用GLM-4.7-flash分析股票"""
        prompt = self._build_analysis_prompt(stock_data, factors)

        payload = {
            "model": "glm-4.7-flash",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的股票分析师，基于技术分析和市场数据提供投资建议。请给出-5到+5的评分（-5强烈卖出，+5强烈买入），置信度0-1，并详细说明分析逻辑。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }

        start_time = time.time()

        try:
            response = self.session.post(self.base_url, json=payload, timeout=30)

            response.raise_for_status()
            result = response.json()

            response_time = time.time() - start_time

            # 解析响应
            content = result["choices"][0]["message"]["content"]
            parsed_result = self._parse_glm_response(content)

            parsed_result.update(
                {
                    "api_source": "glm-4.7-flash",
                    "response_time": response_time,
                    "token_usage": result.get("usage", {}),
                }
            )

            return parsed_result

        except Exception as e:
            print(f"GLM-4.7-flash API调用失败: {e}")
            return self._fallback_analysis(stock_data, factors)

    def _build_analysis_prompt(self, stock_data: Dict, factors: Dict) -> str:
        """构建分析提示"""
        prompt = f"""
请分析以下股票的投资价值：

股票代码: {stock_data.get("stock_code", "")}
当前价格: {stock_data.get("current_price", 0)}元
涨跌幅: {stock_data.get("price_change_pct", 0):.2%}

技术指标:
- 5日动量: {factors.get("momentum_5d", 0):.2%}
- 20日动量: {factors.get("momentum_20d", 0):.2%}
- RSI: {factors.get("rsi", 50):.1f}
- 成交量比率: {factors.get("volume_ratio", 1):.2f}
- 价格位置: {factors.get("price_position", 0.5):.2f}
- 趋势因子: {factors.get("trend_factor", 0):.2f}

请基于以上信息给出：
1. 投资评分（-5到+5）
2. 置信度（0到1）
3. 投资建议（BUY/SELL/HOLD）
4. 风险等级（LOW/MEDIUM/HIGH）
5. 详细分析逻辑

请以JSON格式返回结果：
{{
    "score": 评分,
    "confidence": 置信度,
    "recommendation": "建议",
    "risk_level": "风险等级",
    "reasoning": "详细分析逻辑",
    "key_factors": ["关键因素1", "关键因素2"]
}}
"""
        return prompt

    def _parse_glm_response(self, content: str) -> Dict:
        """解析GLM响应"""
        try:
            # 尝试解析JSON
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                return json.loads(json_str)
        except:
            pass

        # 如果JSON解析失败，使用文本解析
        return self._parse_text_response(content)

    def _parse_text_response(self, content: str) -> Dict:
        """解析文本响应"""
        result = {
            "score": 0,
            "confidence": 0.5,
            "recommendation": "HOLD",
            "risk_level": "MEDIUM",
            "reasoning": content,
            "key_factors": [],
        }

        # 简单的文本解析逻辑
        lines = content.lower().split("\n")

        for line in lines:
            if "评分" in line or "score" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                    result["score"] = np.clip(score, -5, 5)
                except:
                    pass
            elif "置信度" in line or "confidence" in line:
                try:
                    conf = float(line.split(":")[-1].strip())
                    result["confidence"] = np.clip(conf, 0, 1)
                except:
                    pass
            elif "建议" in line or "recommendation" in line:
                if "buy" in line or "买入" in line:
                    result["recommendation"] = "BUY"
                elif "sell" in line or "卖出" in line:
                    result["recommendation"] = "SELL"

        return result

    def _fallback_analysis(self, stock_data: Dict, factors: Dict) -> Dict:
        """回退分析（当API失败时）"""
        # 基于因子计算模拟评分
        momentum_score = factors.get("momentum_5d", 0) * 100
        rsi_score = (50 - factors.get("rsi", 50)) * 0.1
        volume_score = (factors.get("volume_ratio", 1) - 1) * 50
        position_score = (factors.get("price_position", 0.5) - 0.5) * 10

        # 综合评分
        simulated_score = momentum_score + rsi_score + volume_score + position_score
        simulated_score = np.clip(simulated_score, -5, 5)

        # 模拟置信度
        confidence = 0.6 + 0.2 * np.random.random()

        return {
            "score": simulated_score,
            "confidence": confidence,
            "recommendation": "BUY"
            if simulated_score > 1
            else "SELL"
            if simulated_score < -1
            else "HOLD",
            "risk_level": "HIGH" if abs(simulated_score) > 3 else "MEDIUM",
            "reasoning": f"GLM-4.7-flash API调用失败，使用模拟分析 - 动量:{momentum_score:.2f}, RSI:{rsi_score:.2f}, 成交量:{volume_score:.2f}, 位置:{position_score:.2f}",
            "key_factors": ["momentum", "rsi", "volume", "position"],
            "api_source": "glm-4.7-flash-fallback",
        }


def main():
    """主函数 - 测试数据源和GLM模型"""
    print("AI策略数据源和GLM模型测试")
    print("=" * 60)

    # 初始化数据源管理器
    data_manager = DataSourceManager()

    # 初始化GLM客户端
    glm_client = GLMFlashClient()

    # 测试股票池
    print("\n测试股票池加载...")
    stock_pool = data_manager.get_stock_pool()
    if stock_pool:
        test_stock = stock_pool[0]
        print(f"选择测试股票: {test_stock}")

        # 测试数据获取
        print(f"\n测试获取{test_stock}数据...")
        stock_data = data_manager.get_stock_data(test_stock)

        if stock_data is not None and not stock_data.empty:
            print(f"✓ 成功获取{len(stock_data)}条数据")
            print(f"最新数据: {stock_data.iloc[-1]['datetime']}")
            print(f"最新价格: {stock_data.iloc[-1]['close']:.2f}")

            # 计算技术因子
            factors = calculate_technical_factors(stock_data)
            print(f"技术因子: {factors}")

            # 测试GLM分析
            print(f"\n测试GLM-4.7-flash分析...")
            stock_info = {
                "stock_code": test_stock,
                "current_price": stock_data.iloc[-1]["close"],
                "price_change_pct": (
                    stock_data.iloc[-1]["close"] / stock_data.iloc[-2]["close"] - 1
                )
                * 100
                if len(stock_data) > 1
                else 0,
            }

            glm_result = glm_client.analyze_stock(stock_info, factors)

            print(f"\nGLM分析结果:")
            print(f"评分: {glm_result['score']:.2f}")
            print(f"置信度: {glm_result['confidence']:.2%}")
            print(f"建议: {glm_result['recommendation']}")
            print(f"风险: {glm_result['risk_level']}")
            print(f"数据源: {glm_result['api_source']}")

        else:
            print(f"❌ 无法获取{test_stock}数据")
    else:
        print("❌ 无法加载股票池")

    # 关闭连接
    data_manager.close()
    print("\n测试完成")


def calculate_technical_factors(stock_data: pd.DataFrame) -> Dict:
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
        factors["momentum_5d"] = (current_price - close_prices[-6]) / close_prices[-6]
    if len(close_prices) > 20:
        factors["momentum_20d"] = (current_price - close_prices[-21]) / close_prices[
            -21
        ]

    # RSI
    if len(close_prices) > 14:
        delta = np.diff(close_prices[-14:])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain) if len(gain) > 0 else 1
        avg_loss = np.mean(loss) if len(loss) > 0 else 1
        rs = avg_gain / avg_loss if avg_loss > 0 else 1
        factors["rsi"] = 100 - (100 / (1 + rs))

    # 成交量比率
    if len(volumes) > 20:
        factors["volume_ratio"] = volumes[-1] / np.mean(volumes[-20:])

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
        ma_short = np.mean(close_prices[-5:])
        ma_long = np.mean(close_prices[-20:])
        factors["trend_factor"] = (ma_short - ma_long) / ma_long if ma_long > 0 else 0

    return factors


if __name__ == "__main__":
    main()
