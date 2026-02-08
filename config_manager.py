#!/usr/bin/python
"""
数据源配置管理器
自动检测并配置最佳数据源
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


def load_dotenv(env_path):
    """简单的.env文件加载器"""
    if not os.path.exists(env_path):
        return False
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析键值对
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # 移除引号
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                os.environ[key] = value
    
    return True


class DataSourceConfig:
    """数据源配置管理器"""

    def __init__(self, env_file: str = ".env"):
        """初始化配置管理器"""
        self.env_file = env_file
        self.config = {}
        self.available_sources = []

        # 加载环境变量
        self.load_env()

        # 检测可用数据源
        self.detect_sources()

        print("=" * 60)
        print("数据源配置管理器初始化完成")
        print("=" * 60)

    def load_env(self):
        """加载环境变量"""
        env_path = Path(self.env_file)

        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ 已加载环境变量: {self.env_file}")
        else:
            print(f"⚠ 环境变量文件不存在: {self.env_file}")
            print("  将使用默认配置")

        # 读取配置
        self.config = {
            # TQSDK配置
            "tqsdk": {
                "account": os.getenv("TQSDK_ACCOUNT", ""),
                "password": os.getenv("TQSDK_PASSWORD", ""),
                "enabled": bool(
                    os.getenv("TQSDK_ACCOUNT") and os.getenv("TQSDK_PASSWORD")
                ),
            },
            # 通达信配置
            "tdx": {
                "path": os.getenv("TDX_PATH", ""),
                "enabled": os.getenv("TDX_ENABLE", "false").lower() == "true",
            },
            # GLM API配置
            "glm": {
                "api_key": os.getenv("GLM_API_KEY", ""),
                "api_base": os.getenv(
                    "GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"
                ),
                "model": os.getenv("GLM_MODEL", "glm-4"),
                "enabled": bool(os.getenv("GLM_API_KEY")),
            },
            # 其他API配置
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "enabled": bool(os.getenv("OPENAI_API_KEY")),
            },
            "baidu": {
                "api_key": os.getenv("BAIDU_API_KEY", ""),
                "secret_key": os.getenv("BAIDU_SECRET_KEY", ""),
                "model": os.getenv("BAIDU_MODEL", "ernie-bot-4"),
                "enabled": bool(
                    os.getenv("BAIDU_API_KEY") and os.getenv("BAIDU_SECRET_KEY")
                ),
            },
            # 数据源优先级
            "priority": os.getenv("DATA_SOURCE_PRIORITY", "tqsdk,tdx,akshare").split(
                ","
            ),
            # 系统配置
            "system": {
                "enable_cache": os.getenv("ENABLE_CACHE", "true").lower() == "true",
                "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),
                "max_cache_size": int(os.getenv("MAX_CACHE_SIZE", "1000")),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
            },
            # 回测配置
            "backtest": {
                "initial_capital": float(
                    os.getenv("BACKTEST_INITIAL_CAPITAL", "100000")
                ),
                "commission_rate": float(
                    os.getenv("BACKTEST_COMMISSION_RATE", "0.0003")
                ),
            },
        }

    def detect_sources(self):
        """检测可用的数据源"""
        print("\n检测可用数据源...")
        print("-" * 60)

        # 检测TQSDK
        tqsdk_available = self._check_tqsdk()
        if tqsdk_available:
            self.available_sources.append("tqsdk")
            print("✓ TQSDK: 可用")
            print(f"  账户: {self.config['tqsdk']['account']}")
        else:
            print("✗ TQSDK: 不可用")

        # 检测通达信
        tdx_available = self._check_tdx()
        if tdx_available:
            self.available_sources.append("tdx")
            print("✓ 通达信: 可用")
            print(f"  路径: {self.config['tdx']['path']}")
        else:
            print("✗ 通达信: 不可用")

        # 检测AkShare
        akshare_available = self._check_akshare()
        if akshare_available:
            self.available_sources.append("akshare")
            print("✓ AkShare: 可用")
        else:
            print("✗ AkShare: 不可用")

        # 检测GLM API
        if self.config["glm"]["enabled"]:
            print("✓ GLM API: 已配置")
            print(f"  模型: {self.config['glm']['model']}")
        else:
            print("✗ GLM API: 未配置")

        # 检测其他AI API
        if self.config["openai"]["enabled"]:
            print("✓ OpenAI API: 已配置")
        if self.config["baidu"]["enabled"]:
            print("✓ 百度API: 已配置")

        print("-" * 60)
        print(f"可用数据源: {', '.join(self.available_sources) if self.available_sources else '无'}")
        print("=" * 60)

    def _check_tqsdk(self) -> bool:
        """检查TQSDK是否可用"""
        if not self.config["tqsdk"]["enabled"]:
            return False

        try:
            from tqsdk import TqApi, TqAuth

            # 尝试创建连接（不实际连接）
            return True
        except ImportError:
            print("  提示: 需要安装 tqsdk: pip install tqsdk")
            return False
        except Exception as e:
            print(f"  错误: {e}")
            return False

    def _check_tdx(self) -> bool:
        """检查通达信是否可用"""
        if not self.config["tdx"]["enabled"]:
            return False

        tdx_path = self.config["tdx"]["path"]
        if not tdx_path:
            return False

        # 检查路径是否存在
        path = Path(tdx_path)
        if path.exists():
            return True
        else:
            print(f"  路径不存在: {tdx_path}")
            return False

    def _check_akshare(self) -> bool:
        """检查AkShare是否可用"""
        try:
            import akshare as ak

            return True
        except ImportError:
            print("  提示: 需要安装 akshare: pip install akshare")
            return False
        except Exception as e:
            print(f"  错误: {e}")
            return False

    def get_best_source(self) -> Optional[str]:
        """获取最佳数据源"""
        # 按优先级顺序检查
        for source in self.config["priority"]:
            if source in self.available_sources:
                return source

        # 如果没有按优先级找到，返回第一个可用的
        if self.available_sources:
            return self.available_sources[0]

        return None

    def get_tqsdk_config(self) -> Dict:
        """获取TQSDK配置"""
        return self.config["tqsdk"]

    def get_tdx_config(self) -> Dict:
        """获取通达信配置"""
        return self.config["tdx"]

    def get_glm_config(self) -> Dict:
        """获取GLM API配置"""
        return self.config["glm"]

    def get_ai_config(self, provider: str = "glm") -> Dict:
        """获取AI API配置"""
        return self.config.get(provider, {})

    def get_system_config(self) -> Dict:
        """获取系统配置"""
        return self.config["system"]

    def get_backtest_config(self) -> Dict:
        """获取回测配置"""
        return self.config["backtest"]

    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 60)
        print("配置摘要")
        print("=" * 60)

        print("\n【数据源配置】")
        best_source = self.get_best_source()
        print(f"最佳数据源: {best_source or '无'}")
        print(f"可用数据源: {', '.join(self.available_sources) or '无'}")
        print(f"优先级顺序: {' > '.join(self.config['priority'])}")

        print("\n【AI配置】")
        if self.config["glm"]["enabled"]:
            print(f"✓ GLM API: {self.config['glm']['model']}")
        if self.config["openai"]["enabled"]:
            print(f"✓ OpenAI: {self.config['openai']['model']}")
        if self.config["baidu"]["enabled"]:
            print(f"✓ 百度: {self.config['baidu']['model']}")

        print("\n【系统配置】")
        sys_config = self.config["system"]
        print(f"缓存: {'启用' if sys_config['enable_cache'] else '禁用'}")
        print(f"日志级别: {sys_config['log_level']}")

        print("\n【回测配置】")
        bt_config = self.config["backtest"]
        print(f"初始资金: {bt_config['initial_capital']:,.0f}元")
        print(f"手续费率: {bt_config['commission_rate']:.4%}")

        print("=" * 60)


def create_env_template():
    """创建环境变量模板文件"""
    template = """# AI策略系统环境变量配置
# 此文件包含敏感信息，不应上传到Git仓库

# ===== GLM API配置 =====
GLM_API_KEY=your-glm-api-key
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4

# ===== TQSDK配置 =====
TQSDK_ACCOUNT=your-account
TQSDK_PASSWORD=your-password

# ===== 通达信配置 =====
TDX_PATH=/path/to/tdx/vipdoc
TDX_ENABLE=true

# ===== 数据源优先级配置 =====
# 可选值: tqsdk, tdx, akshare, tushare
DATA_SOURCE_PRIORITY=tqsdk,tdx,akshare

# ===== 其他API配置（可选）=====
# OpenAI API
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# 百度文心一言
BAIDU_API_KEY=your-baidu-api-key
BAIDU_SECRET_KEY=your-baidu-secret-key
BAIDU_MODEL=ernie-bot-4

# ===== 系统配置 =====
# 缓存设置
ENABLE_CACHE=true
CACHE_TTL=3600
MAX_CACHE_SIZE=1000

# 日志级别: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# 回测设置
BACKTEST_INITIAL_CAPITAL=100000
BACKTEST_COMMISSION_RATE=0.0003
"""

    env_example_path = Path(".env.example")
    with open(env_example_path, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"✓ 已创建环境变量模板: {env_example_path}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("AI策略系统 - 数据源配置管理器")
    print("=" * 60)

    # 创建配置管理器
    config_manager = DataSourceConfig()

    # 打印配置摘要
    config_manager.print_config_summary()

    # 测试数据源
    best_source = config_manager.get_best_source()
    if best_source:
        print(f"\n✓ 推荐使用数据源: {best_source}")

        if best_source == "tqsdk":
            tqsdk_config = config_manager.get_tqsdk_config()
            print(f"  TQSDK账户: {tqsdk_config['account']}")

        elif best_source == "tdx":
            tdx_config = config_manager.get_tdx_config()
            print(f"  通达信路径: {tdx_config['path']}")

    else:
        print("\n⚠ 警告: 没有可用的数据源！")
        print("  请配置至少一个数据源")

    # 检查AI配置
    glm_config = config_manager.get_glm_config()
    if glm_config["enabled"]:
        print(f"\n✓ GLM API已配置")
        print(f"  模型: {glm_config['model']}")
        print(f"  API地址: {glm_config['api_base']}")
    else:
        print("\n⚠ GLM API未配置")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 检查是否需要创建模板
    if not Path(".env").exists() and not Path(".env.example").exists():
        print("未找到环境变量文件，创建模板...")
        create_env_template()
        print("\n请编辑 .env.example 并重命名为 .env")
        sys.exit(0)

    main()
