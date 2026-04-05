# followin-mcp

一个面向 crypto 场景的 Followin MCP 原型，当前重点是：

- 内容归一化
- 事件聚类
- 用户个性化推荐排序
- 推荐解释与调试

## 目录结构

- `followin_mcp/`
  Python 包入口
- `followin_mcp/core/`
  核心业务逻辑：adapter、model、normalizer、ranking、service
- `followin_mcp/mcp/`
  MCP server 入口
- `followin_mcp/demo/`
  测试 agent 和 Web demo
- `scripts/debug_recommendation.py`
  推荐调试脚本
- `scripts/start_dev.sh`
  本地一键启动脚本
- `web/`
  前端静态资源

## 核心模块

- `followin_mcp/core/adapters.py`
  Followin API 适配层
- `followin_mcp/core/models.py`
  数据模型
- `followin_mcp/core/normalizer.py`
  原始内容标准化、实体抽取、事件类型识别
- `followin_mcp/core/clustering.py`
  事件聚类
- `followin_mcp/core/ranking.py`
  用户推荐排序与解释
- `followin_mcp/core/digest.py`
  简报生成
- `followin_mcp/core/service.py`
  面向 MCP / 应用层的服务入口

## 运行调试脚本

```bash
python3 scripts/debug_recommendation.py
```

## Web Demo

如果你想更直观地测试“随机用户画像 + 多轮对话”，可以启动一个 Web demo：

```bash
python3 -m followin_mcp.demo.webapp
```

或者安装后：

```bash
followin-mcp-web
```

然后打开：

```text
http://127.0.0.1:8000
```

这个 demo 支持：

- 随机生成用户画像
- 为当前画像创建一个 LangChain agent session
- 在同一个 session 里保留多轮聊天上下文
- 展示每轮实际发生的一个或多个 tool 调用
- 展示 tool 参数和返回结果卡片

运行前请确保 `.env` 中已经配置：

```bash
FOLLOWIN_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_api_key
```

## MCP Server

当前已经把以下能力暴露成 MCP tools：

- `get_latest_headlines`
- `get_trending_feeds`
- `get_project_feed`
- `get_project_opinions`
- `get_trending_topics`
- `search_content`

启动方式：

```bash
python3 -m followin_mcp.mcp.server
```

或者安装后使用：

```bash
followin-mcp-server
```
