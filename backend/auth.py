"""用户认证模块 — JWT + bcrypt"""
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
import bcrypt as _bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.database import get_db

# ── 配置 ──────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET", "astock-secret-change-in-production-2026")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 24

# bcrypt 直接调用
bearer_scheme = HTTPBearer(auto_error=False)

# 初始管理员账号（从环境变量读取，默认 admin/admin）
INIT_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
INIT_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")


# ── 数据库操作 ────────────────────────────────────────────────────────────────

async def init_users():
    """初始化用户表，写入默认管理员（若不存在）"""
    async with get_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                hashed_pw  TEXT NOT NULL,
                role       TEXT DEFAULT 'admin',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        # 检查是否已有用户
        async with db.execute("SELECT COUNT(*) FROM users") as cur:
            count = (await cur.fetchone())[0]
        if count == 0:
            hashed = _hash_password(INIT_PASSWORD)
            await db.execute(
                "INSERT INTO users(username, hashed_pw, role) VALUES(?,?,?)",
                (INIT_USERNAME, hashed, "admin")
            )
            await db.commit()
            from loguru import logger
            logger.info(f"默认管理员已创建: {INIT_USERNAME}")


async def get_user(username: str) -> Optional[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT id, username, hashed_pw, role FROM users WHERE username=?", (username,)
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def list_users() -> list[dict]:
    async with get_db() as db:
        async with db.execute("SELECT id, username, role, created_at FROM users ORDER BY id") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def create_user(username: str, password: str, role: str = "admin") -> dict:
    hashed = _hash_password(password)
    async with get_db() as db:
        try:
            await db.execute(
                "INSERT INTO users(username, hashed_pw, role) VALUES(?,?,?)",
                (username, hashed, role)
            )
            await db.commit()
        except Exception:
            raise ValueError(f"用户名 '{username}' 已存在")
    return {"username": username, "role": role}


async def update_password(username: str, new_password: str) -> bool:
    hashed = _hash_password(new_password)
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET hashed_pw=? WHERE username=?", (hashed, username)
        )
        await db.commit()
    return True


async def delete_user(username: str) -> bool:
    async with get_db() as db:
        # 不允许删除最后一个管理员
        async with db.execute("SELECT COUNT(*) FROM users") as cur:
            count = (await cur.fetchone())[0]
        if count <= 1:
            raise ValueError("不能删除最后一个用户")
        await db.execute("DELETE FROM users WHERE username=?", (username,))
        await db.commit()
    return True


# ── 认证逻辑 ──────────────────────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def _hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def create_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> dict:
    """FastAPI 依赖：校验 JWT，返回用户信息"""
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="未登录或 Token 已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not credentials:
        raise exc
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise exc
    except JWTError:
        raise exc
    user = await get_user(username)
    if not user:
        raise exc
    return user
