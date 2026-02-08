#!/usr/bin/python
"""
真实AI API集成模块
集成多种AI服务API，提供真实的AI分析能力替代模拟分析
"""

import numpy as np
import pandas as pd
import requests
import json
import time
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class RealAIApiManager:
    """真实AI API管理器"""

    def __init__(self, config_file: str = "ai_api_config.json"):
        self.config_file = config_file
        self.config = self.load_config()

        # API客户端
        self.api_clients = {}
        self.initialize_api_clients()

        # 缓存管理
        self.cache = {}
        self.cache_ttl = 3600  # 1小时缓存

        # 速率限制
        self.rate_limits = {}
        self.last_request_time = {}

        print("真实AI API管理器初始化完成")

    def load_config(self) -> Dict:
        """加载API配置"""
        default_config = {
            "apis": {
                "openai": {
                    "enabled": True,
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4",
                    "api_key": "your-openai-api-key",
                    "rate_limit": 60,  # 每分钟请求数
                },
                "anthropic": {
                    "enabled": False,
                    "base_url": "https://api.anthropic.com",
                    "model": "claude-3-sonnet-20240229",
                    "api_key": "your-anthropic-api-key",
                    "rate_limit": 50,
                },
                "google": {
                    "enabled": False,
                    "base_url": "https://generativelanguage.googleapis.com",
                    "model": "gemini-pro",
                    "api_key": "your-google-api-key",
                    "rate_limit": 60,
                },
                "baidu": {
                    "enabled": True,
                    "base_url": "https://aip.baidubce.com",
                    "model": "ernie-bot-4",
                    "api_key": "your-baidu-api-key",
                    "secret_key": "your-baidu-secret-key",
                    "rate_limit": 30,
                },
                "alibaba": {
                    "enabled": False,
                    "base_url": "https://dashscope.aliyuncs.com",
                    "model": "qwen-max",
                    "api_key": "your-alibaba-api-key",
                    "rate_limit": 50,
                },
            },
            "fallback": {
                "enable_fallback": True,
                "fallback_to_simulation": True,
                "max_retries": 3,
            },
            "cache": {"enable_cache": True, "cache_ttl": 3600, "max_cache_size": 1000},
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                config = {**default_config, **loaded_config}
                print(f"✓ 已加载API配置: {self.config_file}")
            else:
                config = default_config
                self.save_config(config)
                print(f"✓ 创建默认API配置: {self.config_file}")
        except Exception as e:
            print(f"⚠ API配置加载失败，使用默认配置: {e}")
            config = default_config

        return config

    def save_config(self, config: Dict = None):
        """保存API配置"""
        if config is None:
            config = self.config

        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✓ API配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"⚠ API配置保存失败: {e}")

    def initialize_api_clients(self):
        """初始化API客户端"""
        for api_name, api_config in self.config["apis"].items():
            if api_config.get("enabled", False):
                try:
                    if api_name == "openai":
                        self.api_clients[api_name] = OpenAIClient(api_config)
                    elif api_name == "anthropic":
                        self.api_clients[api_name] = AnthropicClient(api_config)
                    elif api_name == "google":
                        self.api_clients[api_name] = GoogleClient(api_config)
                    elif api_name == "baidu":
                        self.api_clients[api_name] = BaiduClient(api_config)
                    elif api_name == "alibaba":
                        self.api_clients[api_name] = AlibabaClient(api_config)

                    print(f"✓ {api_name} API客户端初始化完成")

                except Exception as e:
                    print(f"⚠ {api_name} API客户端初始化失败: {e}")

    def analyze_stock_with_ai(
        self, stock_data: Dict, factors: Dict, preferred_api: str = "openai"
    ) -> Dict:
        """使用真实AI分析股票"""
        # 生成缓存键
        cache_key = self._generate_cache_key(stock_data, factors)

        # 检查缓存
        if self.config["cache"]["enable_cache"]:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

        # 选择可用的API
        api_client = self._select_available_api(preferred_api)

        if not api_client:
            if self.config["fallback"]["fallback_to_simulation"]:
                print("⚠ 所有AI API不可用，回退到模拟分析")
                return self._simulate_ai_analysis(stock_data, factors)
            else:
                return {"error": "所有AI API不可用且未启用模拟回退"}

        # 准备分析请求
        analysis_request = self._prepare_analysis_request(stock_data, factors)

        # 调用AI API
        try:
            result = api_client.analyze_stock(analysis_request)

            # 处理结果
            processed_result = self._process_ai_result(result, stock_data, factors)

            # 缓存结果
            if self.config["cache"]["enable_cache"]:
                self._save_to_cache(cache_key, processed_result)

            return processed_result

        except Exception as e:
            print(f"⚠ AI API调用失败: {e}")

            # 重试逻辑
            if self.config["fallback"]["max_retries"] > 0:
                return self.analyze_stock_with_ai(
                    stock_data, factors, preferred_api=self._get_next_api(preferred_api)
                )
            else:
                if self.config["fallback"]["fallback_to_simulation"]:
                    return self._simulate_ai_analysis(stock_data, factors)
                else:
                    return {"error": f"AI API调用失败: {str(e)}"}

    def _generate_cache_key(self, stock_data: Dict, factors: Dict) -> str:
        """生成缓存键"""
        # 提取关键数据
        key_data = {
            "stock_code": stock_data.get("stock_code", ""),
            "current_price": stock_data.get("current_price", 0),
            "volume": stock_data.get("volume", 0),
            "timestamp": stock_data.get("timestamp", ""),
            # 取因子的小数点后4位作为特征
            "momentum": round(factors.get("momentum", 0), 4),
            "rsi": round(factors.get("rsi", 0), 4),
            "volume_ratio": round(factors.get("volume_ratio", 0), 4),
            "price_position": round(factors.get("price_position", 0), 4),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存获取结果"""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if (
                time.time() - cached_item["timestamp"]
                < self.config["cache"]["cache_ttl"]
            ):
                return cached_item["result"]
            else:
                del self.cache[cache_key]
        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """保存结果到缓存"""
        self.cache[cache_key] = {"result": result, "timestamp": time.time()}

        # 清理过期缓存
        if len(self.cache) > self.config["cache"]["max_cache_size"]:
            self._cleanup_cache()

    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []

        for key, item in self.cache.items():
            if current_time - item["timestamp"] > self.config["cache"]["cache_ttl"]:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        # 如果还是太大，删除最旧的
        if len(self.cache) > self.config["cache"]["max_cache_size"]:
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1]["timestamp"])
            items_to_remove = len(self.cache) - self.config["cache"]["max_cache_size"]

            for i in range(items_to_remove):
                del self.cache[sorted_items[i][0]]

    def _select_available_api(self, preferred_api: str) -> Optional[Any]:
        """选择可用的API"""
        # 首选API
        if preferred_api in self.api_clients:
            if self._check_rate_limit(preferred_api):
                return self.api_clients[preferred_api]

        # 备选API
        for api_name, client in self.api_clients.items():
            if api_name != preferred_api and self._check_rate_limit(api_name):
                return client

        return None

    def _check_rate_limit(self, api_name: str) -> bool:
        """检查速率限制"""
        api_config = self.config["apis"].get(api_name, {})
        rate_limit = api_config.get("rate_limit", 60)

        current_time = time.time()
        last_request = self.last_request_time.get(api_name, 0)

        # 检查是否在速率限制内
        if current_time - last_request < 60 / rate_limit:
            return False

        self.last_request_time[api_name] = current_time
        return True

    def _prepare_analysis_request(self, stock_data: Dict, factors: Dict) -> Dict:
        """准备AI分析请求"""
        request = {
            "stock_code": stock_data.get("stock_code", ""),
            "stock_name": stock_data.get("stock_name", ""),
            "current_data": {
                "price": stock_data.get("current_price", 0),
                "volume": stock_data.get("volume", 0),
                "high": stock_data.get("high_price", 0),
                "low": stock_data.get("low_price", 0),
                "change": stock_data.get("price_change", 0),
                "change_pct": stock_data.get("price_change_pct", 0),
            },
            "technical_factors": {
                "momentum_5d": factors.get("momentum_5d", 0),
                "momentum_20d": factors.get("momentum_20d", 0),
                "rsi": factors.get("rsi", 50),
                "volume_ratio": factors.get("volume_ratio", 1),
                "price_position": factors.get("price_position", 0.5),
                "trend_factor": factors.get("trend_factor", 0),
                "volatility": factors.get("volatility", 0.02),
            },
            "historical_data": {
                "prices_20d": stock_data.get("prices_20d", []),
                "volumes_20d": stock_data.get("volumes_20d", []),
                "highs_20d": stock_data.get("highs_20d", []),
                "lows_20d": stock_data.get("lows_20d", []),
            },
            "market_context": {
                "market_index": stock_data.get("market_index", 0),
                "market_change": stock_data.get("market_change", 0),
                "sector_performance": stock_data.get("sector_performance", 0),
                "timestamp": datetime.now().isoformat(),
            },
        }

        return request

    def _process_ai_result(
        self, ai_result: Dict, stock_data: Dict, factors: Dict
    ) -> Dict:
        """处理AI分析结果"""
        # 提取AI评分
        ai_score = ai_result.get("score", 0)
        confidence = ai_result.get("confidence", 0.5)
        reasoning = ai_result.get("reasoning", "")

        # 确保评分在有效范围内
        ai_score = np.clip(ai_score, -5, 5)
        confidence = np.clip(confidence, 0, 1)

        # 生成详细分析
        detailed_analysis = {
            "ai_score": ai_score,
            "confidence": confidence,
            "reasoning": reasoning,
            "recommendation": ai_result.get("recommendation", "HOLD"),
            "risk_level": ai_result.get("risk_level", "MEDIUM"),
            "target_price": ai_result.get(
                "target_price", stock_data.get("current_price", 0)
            ),
            "time_horizon": ai_result.get("time_horizon", "SHORT_TERM"),
            "key_factors": ai_result.get("key_factors", []),
            "market_view": ai_result.get("market_view", "NEUTRAL"),
            "api_source": ai_result.get("api_source", "unknown"),
            "response_time": ai_result.get("response_time", 0),
            "token_usage": ai_result.get("token_usage", {}),
            "original_factors": factors,
            "stock_data": {
                "stock_code": stock_data.get("stock_code", ""),
                "current_price": stock_data.get("current_price", 0),
                "timestamp": stock_data.get("timestamp", ""),
            },
        }

        return detailed_analysis

    def _simulate_ai_analysis(self, stock_data: Dict, factors: Dict) -> Dict:
        """模拟AI分析（回退方案）"""
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
            "ai_score": simulated_score,
            "confidence": confidence,
            "reasoning": f"模拟AI分析 - 动量:{momentum_score:.2f}, RSI:{rsi_score:.2f}, 成交量:{volume_score:.2f}, 位置:{position_score:.2f}",
            "recommendation": "BUY"
            if simulated_score > 1
            else "SELL"
            if simulated_score < -1
            else "HOLD",
            "risk_level": "HIGH" if abs(simulated_score) > 3 else "MEDIUM",
            "target_price": stock_data.get("current_price", 0)
            * (1 + simulated_score * 0.05),
            "time_horizon": "SHORT_TERM",
            "key_factors": ["momentum", "rsi", "volume", "position"],
            "market_view": "NEUTRAL",
            "api_source": "simulation",
            "response_time": 0.1,
            "token_usage": {},
            "original_factors": factors,
            "stock_data": {
                "stock_code": stock_data.get("stock_code", ""),
                "current_price": stock_data.get("current_price", 0),
                "timestamp": stock_data.get("timestamp", ""),
            },
        }

    def _get_next_api(self, current_api: str) -> str:
        """获取下一个API"""
        api_list = list(self.config["apis"].keys())
        try:
            current_index = api_list.index(current_api)
            return api_list[(current_index + 1) % len(api_list)]
        except (ValueError, IndexError):
            return api_list[0] if api_list else current_api


class OpenAIClient:
    """OpenAI API客户端"""

    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.api_key = config["api_key"]
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def analyze_stock(self, request_data: Dict) -> Dict:
        """使用OpenAI分析股票"""
        prompt = self._build_analysis_prompt(request_data)

        payload = {
            "model": self.model,
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
            response = self.session.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=30
            )

            response.raise_for_status()
            result = response.json()

            response_time = time.time() - start_time

            # 解析响应
            content = result["choices"][0]["message"]["content"]
            parsed_result = self._parse_openai_response(content)

            parsed_result.update(
                {
                    "api_source": "openai",
                    "response_time": response_time,
                    "token_usage": result.get("usage", {}),
                }
            )

            return parsed_result

        except Exception as e:
            raise Exception(f"OpenAI API调用失败: {str(e)}")

    def _build_analysis_prompt(self, data: Dict) -> str:
        """构建分析提示"""
        prompt = f"""
请分析以下股票的投资价值：

股票代码: {data["stock_code"]}
股票名称: {data["stock_name"]}

当前数据:
- 价格: {data["current_data"]["price"]}
- 涨跌幅: {data["current_data"]["change_pct"]:.2%}
- 成交量: {data["current_data"]["volume"]}

技术指标:
- 5日动量: {data["technical_factors"]["momentum_5d"]:.2%}
- 20日动量: {data["technical_factors"]["momentum_20d"]:.2%}
- RSI: {data["technical_factors"]["rsi"]:.1f}
- 成交量比率: {data["technical_factors"]["volume_ratio"]:.2f}
- 价格位置: {data["technical_factors"]["price_position"]:.2f}
- 趋势因子: {data["technical_factors"]["trend_factor"]:.2f}

市场环境:
- 大盘指数: {data["market_context"]["market_index"]}
- 大盘涨跌: {data["market_context"]["market_change"]:.2%}

请基于以上信息给出：
1. 投资评分（-5到+5）
2. 置信度（0到1）
3. 投资建议（BUY/SELL/HOLD）
4. 风险等级（LOW/MEDIUM/HIGH）
5. 详细分析逻辑
6. 关键影响因素

请以JSON格式返回结果：
{{
    "score": 评分,
    "confidence": 置信度,
    "recommendation": "建议",
    "risk_level": "风险等级",
    "reasoning": "详细分析逻辑",
    "key_factors": ["关键因素1", "关键因素2"],
    "target_price": 目标价格,
    "time_horizon": "时间周期",
    "market_view": "市场观点"
}}
"""
        return prompt

    def _parse_openai_response(self, content: str) -> Dict:
        """解析OpenAI响应"""
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
        # 默认值
        result = {
            "score": 0,
            "confidence": 0.5,
            "recommendation": "HOLD",
            "risk_level": "MEDIUM",
            "reasoning": content,
            "key_factors": [],
            "target_price": 0,
            "time_horizon": "SHORT_TERM",
            "market_view": "NEUTRAL",
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


class BaiduClient:
    """百度文心一言API客户端"""

    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config["api_key"]
        self.secret_key = config["secret_key"]
        self.model = config["model"]
        self.access_token = None
        self.token_expires = 0

        self.session = requests.Session()
        self._get_access_token()

    def _get_access_token(self):
        """获取访问令牌"""
        current_time = time.time()

        if self.access_token and current_time < self.token_expires:
            return

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }

        try:
            response = self.session.post(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()

            self.access_token = result["access_token"]
            self.token_expires = (
                current_time + result["expires_in"] - 300
            )  # 提前5分钟刷新

        except Exception as e:
            raise Exception(f"百度API获取访问令牌失败: {str(e)}")

    def analyze_stock(self, request_data: Dict) -> Dict:
        """使用百度文心一言分析股票"""
        self._get_access_token()

        prompt = self._build_analysis_prompt(request_data)

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "top_p": 0.8,
            "penalty_score": 1.0,
            "stream": False,
        }

        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model}?access_token={self.access_token}"

        start_time = time.time()

        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            response_time = time.time() - start_time

            if "error_code" in result:
                raise Exception(f"百度API错误: {result['error_msg']}")

            # 解析响应
            content = result["result"]
            parsed_result = self._parse_baidu_response(content)

            parsed_result.update(
                {
                    "api_source": "baidu",
                    "response_time": response_time,
                    "token_usage": {
                        "total_tokens": result.get("usage", {}).get("total_tokens", 0)
                    },
                }
            )

            return parsed_result

        except Exception as e:
            raise Exception(f"百度API调用失败: {str(e)}")

    def _build_analysis_prompt(self, data: Dict) -> str:
        """构建分析提示"""
        prompt = f"""作为专业股票分析师，请分析以下股票：

股票：{data["stock_code"]} ({data["stock_name"]})
当前价格：{data["current_data"]["price"]}元
涨跌幅：{data["current_data"]["change_pct"]:.2%}

技术分析：
- 5日动量：{data["technical_factors"]["momentum_5d"]:.2%}
- 20日动量：{data["technical_factors"]["momentum_20d"]:.2%}
- RSI指标：{data["technical_factors"]["rsi"]:.1f}
- 成交量比率：{data["technical_factors"]["volume_ratio"]:.2f}
- 价格位置：{data["technical_factors"]["price_position"]:.2f}

请给出投资评分（-5强烈卖出到+5强烈买入）、置信度（0-1）、具体建议和风险分析。"""

        return prompt

    def _parse_baidu_response(self, content: str) -> Dict:
        """解析百度响应"""
        # 类似OpenAI的解析逻辑
        try:
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                return json.loads(json_str)
        except:
            pass

        # 文本解析
        result = {
            "score": 0,
            "confidence": 0.5,
            "recommendation": "HOLD",
            "risk_level": "MEDIUM",
            "reasoning": content,
            "key_factors": [],
            "target_price": 0,
            "time_horizon": "SHORT_TERM",
            "market_view": "NEUTRAL",
        }

        # 简单解析逻辑
        content_lower = content.lower()

        # 提取评分
        import re

        score_match = re.search(r"评分[：:]\s*([+-]?\d+\.?\d*)", content)
        if score_match:
            try:
                score = float(score_match.group(1))
                result["score"] = np.clip(score, -5, 5)
            except:
                pass

        # 提取置信度
        conf_match = re.search(r"置信度[：:]\s*(\d+\.?\d*)", content)
        if conf_match:
            try:
                conf = float(conf_match.group(1))
                result["confidence"] = np.clip(conf, 0, 1)
            except:
                pass

        # 提取建议
        if any(word in content_lower for word in ["买入", "建议买入", "buy"]):
            result["recommendation"] = "BUY"
        elif any(word in content_lower for word in ["卖出", "建议卖出", "sell"]):
            result["recommendation"] = "SELL"

        return result


# 其他API客户端的简化实现
class AnthropicClient:
    """Anthropic Claude API客户端"""

    def __init__(self, config: Dict):
        self.config = config

    def analyze_stock(self, request_data: Dict) -> Dict:
        """Anthropic API分析（简化实现）"""
        return {
            "score": 1.5,
            "confidence": 0.7,
            "reasoning": "Anthropic Claude分析结果（待实现）",
            "api_source": "anthropic",
        }


class GoogleClient:
    """Google Gemini API客户端"""

    def __init__(self, config: Dict):
        self.config = config

    def analyze_stock(self, request_data: Dict) -> Dict:
        """Google Gemini API分析（简化实现）"""
        return {
            "score": 1.2,
            "confidence": 0.6,
            "reasoning": "Google Gemini分析结果（待实现）",
            "api_source": "google",
        }


class AlibabaClient:
    """阿里巴巴通义千问API客户端"""

    def __init__(self, config: Dict):
        self.config = config

    def analyze_stock(self, request_data: Dict) -> Dict:
        """阿里巴巴API分析（简化实现）"""
        return {
            "score": 1.8,
            "confidence": 0.65,
            "reasoning": "阿里巴巴通义千问分析结果（待实现）",
            "api_source": "alibaba",
        }


def main():
    """主函数 - 测试AI API集成"""
    print("真实AI API集成测试")
    print("=" * 60)

    # 创建AI API管理器
    ai_manager = RealAIApiManager()

    # 模拟股票数据
    stock_data = {
        "stock_code": "000001",
        "stock_name": "平安银行",
        "current_price": 12.50,
        "volume": 1000000,
        "price_change": 0.25,
        "price_change_pct": 0.02,
        "timestamp": datetime.now().isoformat(),
    }

    # 模拟因子数据
    factors = {
        "momentum_5d": 0.03,
        "momentum_20d": 0.08,
        "rsi": 55,
        "volume_ratio": 1.2,
        "price_position": 0.6,
        "trend_factor": 0.02,
    }

    print(f"\n分析股票: {stock_data['stock_name']} ({stock_data['stock_code']})")
    print(f"当前价格: {stock_data['current_price']}元")

    # 调用AI分析
    print("\n正在进行AI分析...")
    result = ai_manager.analyze_stock_with_ai(stock_data, factors)

    # 显示结果
    print("\n" + "=" * 60)
    print("AI分析结果")
    print("=" * 60)

    if "error" in result:
        print(f"❌ 分析失败: {result['error']}")
    else:
        print(f"🤖 AI评分: {result['ai_score']:.2f}/5.0")
        print(f"📊 置信度: {result['confidence']:.2%}")
        print(f"💡 投资建议: {result['recommendation']}")
        print(f"⚠️  风险等级: {result['risk_level']}")
        print(f"🎯 目标价格: {result['target_price']:.2f}元")
        print(f"⏰ 时间周期: {result['time_horizon']}")
        print(f"🔍 数据源: {result['api_source']}")
        print(f"⚡ 响应时间: {result['response_time']:.2f}秒")

        print(f"\n📝 分析逻辑:")
        print(result["reasoning"])

        if result.get("key_factors"):
            print(f"\n🔑 关键因素: {', '.join(result['key_factors'])}")

    print("=" * 60)


if __name__ == "__main__":
    main()
