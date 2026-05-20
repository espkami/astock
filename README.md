# A股新闻·龙头股匹配系统

基于 FastAPI + SQLite + 大模型的 A 股新闻采集与龙头股匹配系统。

## 功能

- **多路新闻采集**：NewsAPI.ai / RSS / 大模型搜索 / 百度·微博·抖音热搜
- **大模型分类**：自动提取摘要、情感（利好/利空/中性）、行业标签、关键词
- **A股龙头匹配**：四阶段流水线将新闻与 A 股龙头股关联
- **多模型轮询**：配置多个大模型账户，429 时自动切换
- **定时任务**：采集/匹配支持定时调度
- **数据保留**：可配置保留天数，自动清理历史数据

## 快速启动

### Docker（推荐）

```bash
# 普通启动
docker-compose up -d

# 调试模式（debugpy:5678）
DEBUG=1 docker-compose up -d
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
│   ├── main.py              # FastAPI 入口
│   ├── database.py          # SQLite 初始化
│   ├── auth.py              # JWT 认证
│   ├── models.py            # Pydantic 模型
│   ├── services/
│   │   ├── news_collector.py   # 多路采集（NewsAPI/RSS/热搜/大模型）
│   │   ├── news_processor.py   # 大模型分类
│   │   ├── matcher.py          # 龙头股匹配引擎
│   │   ├── llm_client.py       # 多模型轮询客户端
│   │   ├── scheduler.py        # APScheduler 定时任务
│   │   ├── stock_service.py    # 股票数据服务
│   │   └── config_service.py   # 配置读写（带缓存）
│   └── routers/             # 路由模块
├── frontend/
│   ├── index.html           # SPA 主文件
│   ├── styles.css           # Obvious 设计系统
│   └── theme.js             # 深色/浅色主题
├── data/                    # SQLite 数据库（运行时生成）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DB_PATH` | `/app/data/astock.db` | 数据库路径 |
| `ADMIN_USERNAME` | `admin` | 初始管理员账号 |
| `ADMIN_PASSWORD` | `admin` | 初始管理员密码 |
| `JWT_SECRET` | `astock-secret-...` | JWT 签名密钥（生产环境请修改） |
| `DEBUG` | `0` | `1` 启用 debugpy 调试模式 |

## 大模型配置

在「⚙️ 设置 → 大模型配置」页面配置，支持：
- Anthropic (Claude)
- OpenAI (GPT)
- 阿里云 Qwen
- 智谱 GLM
- DeepSeek
- 自定义（兼容 OpenAI API 格式）

可添加多个账户，系统自动轮询，遇到 429 限速自动切换。
