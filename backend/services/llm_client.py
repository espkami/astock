"""大模型统一客户端 — 多 Provider 路由"""
import json
import asyncio
from typing import Any, Optional
from loguru import logger

# 当前使用的 Key 索引（轮询用）
_key_index: dict = {}  # {provider_type: int}
# 最近一次故障切换消息（供进度显示用）
_last_failover_msg: list = [""]  # 用list以便引用传递

def get_last_failover_msg() -> str:
    """获取并清除最近一次故障切换消息"""
    msg = _last_failover_msg[0]
    _last_failover_msg[0] = ""
    return msg


PROVIDERS = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4", "glm-4-flash", "glm-4-air"],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-coder"],
    },
}


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str, base_url: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or PROVIDERS.get(provider, {}).get("base_url", "")

    @staticmethod
    def _default_base_url(provider: str) -> str:
        return PROVIDERS.get(provider, {}).get("base_url", "")

    async def chat(self, messages: list[dict], json_mode: bool = False, timeout: int = 30) -> str:
        """统一对话接口，返回文本内容。
        故障转移策略：遇到 429/5xx/超时，自动切换到下一个已开启的模型，
        轮完所有模型仍失败才抛出异常。
        """
        from backend.services.config_service import get_config as _gc
        # 获取当前所有开启的模型列表（用于故障转移）
        models_cfg = await _gc("llm_models", [])
        active = [m for m in models_cfg if m.get("enabled") is not False
                  and m.get("provider") and m.get("api_key") and m.get("model")]
        if not active:
            active = [{"provider": self.provider, "api_key": self.api_key,
                       "model": self.model, "base_url": self.base_url}]

        # 找当前模型在列表中的位置，作为起点
        try:
            start_idx = next(i for i, m in enumerate(active)
                             if m["api_key"] == self.api_key and m["model"] == self.model)
        except StopIteration:
            start_idx = 0

        last_err = None
        for attempt in range(len(active)):
            idx = (start_idx + attempt) % len(active)
            m = active[idx]
            # 当次尝试用的配置存入局部变量，不修改 self，避免污染下次调用的起点
            cur_provider = m["provider"]
            cur_api_key  = m["api_key"]
            cur_model    = m["model"]
            cur_base_url = m.get("base_url") or LLMClient._default_base_url(cur_provider)
            if attempt > 0:
                logger.info(f"LLM 故障转移 → {cur_provider}/{cur_model} ({cur_api_key[:8]}...)")
                await asyncio.sleep(1)
            try:
                if cur_provider == "anthropic":
                    return await self._call_anthropic_with(messages, json_mode, timeout,
                                                           cur_api_key, cur_model)
                else:
                    return await self._call_openai_compat_with(messages, json_mode, timeout,
                                                               cur_api_key, cur_model, cur_base_url)
            except Exception as e:
                err_str = str(e)
                last_err = e
                # 400 内容审核拒绝：不是模型故障，直接抛出不转移
                if "400" in err_str:
                    logger.warning(f"LLM 内容被拒(400)，跳过: {err_str[:80]}")
                    raise
                # 判断是否值得转移：429限速 / 5xx服务错误 / 超时
                is_failover = (
                    "429" in err_str or
                    "500" in err_str or "502" in err_str or "503" in err_str or
                    "timeout" in err_str.lower() or "timed out" in err_str.lower()
                )
                if is_failover and attempt < len(active) - 1:
                    logger.warning(f"LLM [{m['provider']}/{m['model']}] 故障({err_str[:60]})，切换下一个模型")
                    # 记录切换信息供进度回调使用
                    next_m = active[(start_idx + attempt + 1) % len(active)]
                    _last_failover_msg[0] = f"⚡ 模型故障切换 → {next_m['provider']}/{next_m['model']}"
                    continue
                elif not is_failover and attempt == 0:
                    # 非故障类错误，原模型重试一次
                    logger.warning(f"LLM 调用失败，重试: {e}")
                    await asyncio.sleep(1)
                    try:
                        if cur_provider == "anthropic":
                            return await self._call_anthropic_with(messages, json_mode, timeout,
                                                                   cur_api_key, cur_model)
                        else:
                            return await self._call_openai_compat_with(messages, json_mode, timeout,
                                                                       cur_api_key, cur_model, cur_base_url)
                    except Exception as e2:
                        last_err = e2
                        logger.error(f"LLM 调用最终失败: {e2}")
                        raise
                else:
                    logger.error(f"LLM 所有模型均失败，最后错误: {err_str[:100]}")
                    raise
        logger.error(f"LLM 所有 {len(active)} 个模型均故障，放弃: {last_err}")
        raise last_err

    async def _call_anthropic(self, messages: list[dict], json_mode: bool, timeout: int) -> str:
        return await self._call_anthropic_with(messages, json_mode, timeout, self.api_key, self.model)

    async def _call_anthropic_with(self, messages: list[dict], json_mode: bool, timeout: int,
                                    api_key: str, model: str) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 2048,
            "messages": user_msgs,
        }
        if system_msg:
            kwargs["system"] = system_msg
        resp = await asyncio.wait_for(client.messages.create(**kwargs), timeout=timeout)
        if hasattr(resp, "usage") and resp.usage:
            asyncio.ensure_future(_record_token_usage(
                getattr(resp.usage, "input_tokens", 0),
                getattr(resp.usage, "output_tokens", 0),
            ))
        return resp.content[0].text

    async def _call_openai_compat(self, messages: list[dict], json_mode: bool, timeout: int) -> str:
        return await self._call_openai_compat_with(messages, json_mode, timeout,
                                                    self.api_key, self.model, self.base_url)

    async def _call_openai_compat_with(self, messages: list[dict], json_mode: bool, timeout: int,
                                        api_key: str, model: str, base_url: str) -> str:
        import httpx
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.3,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            if usage:
                asyncio.ensure_future(_record_token_usage(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                ))
            return data["choices"][0]["message"]["content"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """获取文本嵌入向量，自动分批（避免超出接口单次上限），失败时返回空列表"""
        try:
            if self.provider == "openai" or self.provider in ("gitee", "custom", ""):
                return await self._embed_openai_batched(texts)
            elif self.provider == "qwen":
                return await self._embed_qwen(texts)
            else:
                return await self._embed_openai_batched(texts)
        except Exception as e:
            logger.warning(f"Embedding 失败，将使用 TF-IDF: {e}")
            return []

    async def _embed_openai_batched(self, texts: list[str], batch_size: int = 24) -> list[list[float]]:
        """分批调用 OpenAI 兼容 Embedding 接口（Gitee 上限 24 条/批）"""
        import httpx, asyncio as _asyncio
        all_vecs = []
        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "X-Failover-Enabled": "true",
                    },
                    json={"model": self.model or "text-embedding-3-small", "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                all_vecs.extend([d["embedding"] for d in data])
                if i + batch_size < len(texts):
                    await _asyncio.sleep(0.2)
        return all_vecs

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_openai_batched(texts)

    async def _embed_qwen(self, texts: list[str]) -> list[list[float]]:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Failover-Enabled": "true",
                },
                json={"model": self.model or "text-embedding-v3", "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]


async def _record_token_usage(input_tokens: int, output_tokens: int):
    """累计写入 token 消耗到 config 表，按月滚动重置"""
    try:
        from backend.database import get_db
        from datetime import datetime, timezone
        import json as _json
        this_month = datetime.now(timezone.utc).strftime("%Y-%m")
        async with get_db() as db:
            async with db.execute(
                "SELECT value FROM config WHERE key='token_usage_total'"
            ) as cur:
                row = await cur.fetchone()
            current = _json.loads(row[0]) if row else {}
            # 月份变了就自动重置，开始新一个月的计数
            if current.get("month") != this_month:
                current = {"month": this_month, "input": 0, "output": 0, "calls": 0}
            current["input"]  = current.get("input",  0) + input_tokens
            current["output"] = current.get("output", 0) + output_tokens
            current["calls"]  = current.get("calls",  0) + 1
            val = _json.dumps(current)
            if row:
                await db.execute("UPDATE config SET value=? WHERE key='token_usage_total'", (val,))
            else:
                await db.execute("INSERT INTO config(key,value) VALUES('token_usage_total',?)", (val,))
            await db.commit()
    except Exception as e:
        logger.warning(f"token 计数写入失败: {e}")


async def _rotate_to_next_model(client: "LLMClient") -> bool:
    """已由 chat() 内部故障转移逻辑替代，保留此函数仅供兼容"""
    return False


async def get_llm_client() -> Optional[LLMClient]:
    """从多模型列表轮询取下一个可用配置"""
    from backend.services.config_service import get_config

    # 优先读 llm_models 列表
    models_cfg = await get_config("llm_models", [])
    active = [m for m in models_cfg if m.get("enabled") is not False
              and m.get("provider") and m.get("api_key") and m.get("model")]

    # fallback：兼容旧的单模型配置
    if not active:
        provider = await get_config("llm_provider")
        api_key  = await get_config("llm_api_key")
        model    = await get_config("llm_model")
        base_url = await get_config("llm_base_url")
        if provider and api_key and model:
            active = [{"provider": provider, "api_key": api_key,
                       "model": model, "base_url": base_url or ""}]

    if not active:
        return None

    # 轮询索引
    idx_key = "llm_models"
    current_idx = _key_index.get(idx_key, 0) % len(active)
    _key_index[idx_key] = (current_idx + 1) % len(active)
    m = active[current_idx]

    return LLMClient(
        provider=m["provider"],
        api_key=m["api_key"],
        model=m["model"],
        base_url=m.get("base_url") or LLMClient._default_base_url(m["provider"]),
    )


async def get_embed_client() -> Optional["LLMClient"]:
    """获取 Embedding 专用模型客户端（优先读 embed_models 列表）"""
    from backend.services.config_service import get_config

    # 优先读列表配置（新版）
    embed_models = await get_config("embed_models", [])
    active = [m for m in embed_models if m.get("enabled") is not False
              and m.get("api_key") and m.get("model")]

    # 兼容旧版单字段配置
    if not active:
        provider = await get_config("embed_provider", "")
        api_key  = await get_config("embed_api_key", "")
        model    = await get_config("embed_model", "")
        base_url = await get_config("embed_base_url", "")
        if provider and api_key and model:
            active = [{"provider": provider, "api_key": api_key,
                       "model": model, "base_url": base_url}]

    if active:
        m = active[0]
        provider = m.get("provider", "openai")
        # custom/gitee/ollama 内部走 openai 兼容路径
        actual_provider = "openai" if provider in ("custom", "gitee", "ollama") else provider
        base_url = m.get("base_url", "") or LLMClient._default_base_url(actual_provider)
        return LLMClient(
            provider=actual_provider,
            api_key=m["api_key"],
            model=m["model"],
            base_url=base_url,
        )

    # fallback：从分析匹配模型里找支持 embedding 的（openai / qwen）
    models_cfg = await get_config("llm_models", [])
    llm_active = [m for m in models_cfg if m.get("enabled") is not False
                  and m.get("provider") and m.get("api_key") and m.get("model")]
    for m in llm_active:
        if m["provider"] in ("openai", "qwen"):
            logger.debug(f"Embedding fallback 到分析匹配模型: {m['provider']}/{m['model']}")
            return LLMClient(
                provider=m["provider"],
                api_key=m["api_key"],
                model=m["model"],
                base_url=m.get("base_url") or LLMClient._default_base_url(m["provider"]),
            )

    logger.warning("未找到支持 Embedding 的模型，将使用 TF-IDF")
    return None


async def test_llm_connection(provider: str, api_key: str, model: str, base_url: Optional[str] = None) -> dict:
    """测试 API Key 连通性"""
    client = LLMClient(provider=provider, api_key=api_key, model=model, base_url=base_url)
    try:
        result = await client.chat([{"role": "user", "content": "Reply with: ok"}], timeout=15)
        return {"success": True, "response": result[:100]}
    except Exception as e:
        return {"success": False, "error": str(e)}
