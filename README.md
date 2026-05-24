# A股新闻·龙头股匹配系统

基于 FastAPI + SQLite + 大模型的 A 股新闻采集与龙头股匹配系统。

## 功能

- **多路新闻采集**：NewsAPI.ai / RSS / 大模型搜索 / 百度·微博·抖音热搜
- **大模型分类**：自动提取摘要、情感（利好/利空/中性）、行业标签、关键词（词根提取，可直接匹配股票标签）
- **四阶段匹配流水线**：标签粗筛 → Embedding 语义排序 → 大模型精排 → top-k 输出
- **股票标签体系**：5500+ 只股票主营业务标签（stock_tags）+ 概念板块标签（stock_board_tags）
- **多模型轮询**：配置多个大模型账户，429 时自动切换；Embedding 模型独立配置
- **灵活匹配策略**：行业龙头 / 直接相关 / 产业链 / 广泛行业，前台实时切换
- **定时任务**：采集/匹配支持定时调度
- **数据保留**：可配置保留天数，自动清理历史数据

## 快速启动

### Docker（推荐）

```bash
docker-compose up -d
```

访问 http://localhost:8000，默认账号 admin / admin。

### 本地启动

```bash
pip install -r requirements.txt
DB_PATH=./data/astock.db uvicorn backend.main:app --host 0.0.0.0 --port 8000 --app-dir .
```

## 目录结构

```
astock/
├── backend/
│   ├── main.py                  # FastAPI 入口 + match-now 单条匹配接口
│   ├── database.py              # SQLite 初始化
│   ├── auth.py                  # JWT 认证
│   ├── models.py                # Pydantic 模型
│   ├── services/
│   │   ├── news_collector.py    # 多路采集（NewsAPI/RSS/热搜/大模型）
│   │   ├── news_processor.py    # 大模型分类（keywords 词根提取）
│   │   ├── matcher.py           # 四阶段匹配引擎
│   │   ├── tag_service.py       # 股票主营业务标签生成（AKShare + LLM）
│   │   ├── board_tag_service.py # 概念/行业板块标签生成
│   │   ├── llm_client.py        # 多模型轮询客户端（含 Embedding）
│   │   ├── scheduler.py         # APScheduler 定时任务
│   │   ├── stock_service.py     # 股票数据服务
│   │   └── config_service.py    # 配置读写（带缓存）
│   └── routers/                 # 路由模块
├── frontend/
│   ├── index.html               # SPA 主文件
│   ├── styles.css               # 样式
│   └── theme.js                 # 深色/浅色主题
├── data/                        # SQLite 数据库（运行时生成）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 匹配流水线

```
新闻 → LLM分析（keywords词根/industries/sentiment）
     → 标签粗筛（stock_tags子串匹配 + board_tags + 名称 + profile描述）
     → Embedding语义排序（NVIDIA nv-embed-v1，fallback TF-IDF）
     → LLM精排（top_k×3候选 → top_k结果）
     → 写入 match_results
```

## 配置项说明

以下配置均存于 DB，运行时实时读取，**无需重启服务**：

| 配置键 | 说明 | 默认值 |
|--------|------|--------|
| `llm_models` | 大模型列表，轮询调用 | — |
| `embed_models` | Embedding 模型列表，取第一个启用的 | — |
| `analysis_focus` | 新闻分析侧重（balanced/stock/industry/macro） | balanced |
| `sentiment_standard` | 情感判断松紧（strict/normal/sensitive） | normal |
| `match_company_type` | 匹配策略（leader/direct/chain/broad） | leader |
| `match_top_k` | 每条新闻最多返回股票数 | 5 |

> **注意**：`match_company_type` 和 `match_top_k` 在前台页面仅在用户主动操作（拨动滑条/点击单选）时触发 onchange 写入 DB，页面初始加载不会写入。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DB_PATH` | `/app/data/astock.db` | 数据库路径 |
| `ADMIN_USERNAME` | `admin` | 初始管理员账号 |
| `ADMIN_PASSWORD` | `admin` | 初始管理员密码 |
| `JWT_SECRET` | `astock-secret-...` | JWT 签名密钥（生产环境请修改） |
| `DEBUG` | `0` | `1` 启用 debugpy 调试模式 |

## 大模型支持

在「⚙️ 设置 → 大模型配置」页面配置，支持：
- 智谱 GLM
- 阿里云 Qwen
- OpenAI (GPT)
- Anthropic (Claude)
- DeepSeek
- 自定义（兼容 OpenAI API 格式，如 NVIDIA NIM、本地 Ollama）

Embedding 模型在「Embedding 模型」标签页独立配置，当前推荐使用 NVIDIA `nv-embed-v1`（4096维）。
