# AI策略系统配置说明

## 快速开始

### 1. 配置API密钥

复制示例配置文件：
```bash
cp ai_api_config.example.json ai_api_config.json
```

编辑 `ai_api_config.json`，填入你的真实API密钥。

### 2. 支持的AI服务

#### OpenAI GPT-4
- 获取API Key: https://platform.openai.com/api-keys
- 配置项: `apis.openai.api_key`

#### Anthropic Claude
- 获取API Key: https://console.anthropic.com/
- 配置项: `apis.anthropic.api_key`

#### Google Gemini
- 获取API Key: https://makersuite.google.com/app/apikey
- 配置项: `apis.google.api_key`

#### 百度文心一言
- 获取API Key: https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application
- 配置项: `apis.baidu.api_key` 和 `apis.baidu.secret_key`

#### 阿里巴巴通义千问
- 获取API Key: https://dashscope.console.aliyun.com/apiKey
- 配置项: `apis.alibaba.api_key`

#### 智谱AI GLM-4
- 获取API Key: https://open.bigmodel.cn/usercenter/apikeys
- 配置项: `apis.zhipu.api_key`

### 3. 配置示例

```json
{
  "apis": {
    "zhipu": {
      "enabled": true,
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "model": "glm-4",
      "api_key": "你的GLM API Key",
      "rate_limit": 60
    }
  }
}
```

### 4. 环境变量方式（推荐）

也可以使用环境变量配置（更安全）：

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key-here"
$env:ZHIPU_API_KEY="your-glm-key-here"
$env:BAIDU_API_KEY="your-baidu-key-here"
$env:BAIDU_SECRET_KEY="your-baidu-secret-here"

# Linux/Mac
export OPENAI_API_KEY="your-key-here"
export ZHIPU_API_KEY="your-glm-key-here"
export BAIDU_API_KEY="your-baidu-key-here"
export BAIDU_SECRET_KEY="your-baidu-secret-here"
```

### 5. 安全注意事项

⚠️ **重要提醒**：
- ✅ `ai_api_config.json` 已添加到 `.gitignore`，不会被提交到Git
- ✅ 只提交 `ai_api_config.example.json` 示例文件
- ❌ 永远不要将真实API Key提交到代码仓库
- ❌ 不要在代码中硬编码API Key
- ✅ 使用环境变量或配置文件管理密钥
- ✅ 定期轮换API密钥

### 6. 配置优先级

系统按以下优先级读取配置：
1. 环境变量（最高优先级）
2. `ai_api_config.json` 配置文件
3. 代码中的默认配置（最低优先级）

### 7. 测试配置

运行测试脚本验证配置：
```bash
python real_ai_api.py
```

### 8. 回退机制

如果所有AI API都不可用，系统会自动回退到模拟分析模式，确保策略正常运行。

## 配置文件结构

```json
{
  "apis": {
    "服务名称": {
      "enabled": true/false,      // 是否启用
      "base_url": "API基础URL",
      "model": "模型名称",
      "api_key": "API密钥",
      "rate_limit": 60            // 每分钟请求限制
    }
  },
  "fallback": {
    "enable_fallback": true,      // 启用回退机制
    "fallback_to_simulation": true, // 回退到模拟分析
    "max_retries": 3              // 最大重试次数
  },
  "cache": {
    "enable_cache": true,         // 启用缓存
    "cache_ttl": 3600,            // 缓存有效期（秒）
    "max_cache_size": 1000        // 最大缓存条目
  }
}
```

## 故障排查

### API调用失败
1. 检查API Key是否正确
2. 检查网络连接
3. 查看API服务状态
4. 检查速率限制

### 配置文件未找到
1. 确认 `ai_api_config.json` 存在
2. 检查文件路径
3. 使用示例文件创建配置

### 环境变量未生效
1. 重启终端/IDE
2. 检查变量名拼写
3. 确认变量已设置：`echo $env:ZHIPU_API_KEY`（Windows）或 `echo $ZHIPU_API_KEY`（Linux/Mac）
