"""Owner commands: ``!colab on|off|status`` and ``!whoami``.

``!colab`` is for the ids in ``LINE_ADMIN_USER_IDS`` only; anyone else gets
「此指令僅限管理員」. It works in 1:1 chats and, like ``!new``, as a group
trigger on its own (``!colab status`` needs no mention). ``!whoami`` answers
the sender's LINE userId in a 1:1 chat only, so an owner can find the id
to put in ``LINE_ADMIN_USER_IDS``; in a group it would show the id to
every member, so there it only says to ask in private.

- ``on``: ``desired=auto``; a DISABLED or COOLDOWN session goes back to
  IDLE (failures and the cooldown cleared), and ``manual_demand_at`` is set,
  so the controller starts a session on its next tick under either
  MODEL_PATH_PREFERENCE (``api`` starts sessions only this way). The idle
  clock counts from the later of this command, the last triggered message
  and READY, so a session always gets a full COLAB_IDLE_TTL_SECONDS.
- ``off``: ``desired=off``; an ASSIGNING, BOOTING or READY session goes to
  DRAINING at once (its tiers stop being used, a late ``ready`` from its VM
  is refused), and the controller releases it.
- ``status`` (or a bare ``!colab``): the session in a few lines (a READY
  session whose router is not loaded or not wholly in VRAM says it routes
  without it; a model whose tag moved since its digest was verified is
  named, for information).

The commands answer with a reply (free) and skip the rate limit, the
router and the models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.services.colab import protocol
from src.services.colab.state import (
    ASSIGNING,
    AUTO,
    BOOTING,
    COOLDOWN,
    DISABLED,
    DRAINING,
    IDLE,
    OFF,
    READY,
    ColabState,
    ColabStateError,
)
from src.utils.logger import logger

ADMIN_ONLY = "此指令僅限管理員"
WHOAMI_DM_ONLY = "!whoami 只在私訊回覆，避免在群組公開你的 userId；請私訊我再試一次。"
COLAB_OFF_CONFIG = "Colab 層未啟用（COLAB_ENABLED=false 或設定不完整，原因見 /health/detail）。"
COLAB_USAGE = "用法：!colab on｜!colab off｜!colab status"
STATE_UNAVAILABLE = "暫時讀不到 Colab 狀態，請稍後再試。"

_COLAB = re.compile(r"!colab(?:\s+(.*))?", re.IGNORECASE | re.DOTALL)
_WHOAMI = re.compile(r"!whoami\s*", re.IGNORECASE)
# What makes a group message a trigger (webhook_handler.should_handle).
GROUP_TRIGGER = re.compile(r"^(?:!hej\s*)?!colab(?:\s|$)", re.IGNORECASE)
_TIMEZONE = ZoneInfo("Asia/Taipei")
_STATUS_NAMES = {
    IDLE: "IDLE（未開機）",
    ASSIGNING: "ASSIGNING（申請 runtime 中）",
    BOOTING: "BOOTING（開機中）",
    READY: "READY（可使用）",
    DRAINING: "DRAINING（釋放中）",
    COOLDOWN: "COOLDOWN（冷卻中）",
    DISABLED: "DISABLED（已停用）",
}


@dataclass(frozen=True)
class Command:
    name: str  # "colab" | "whoami"
    arg: str = ""


def parse_command(text: str) -> Command | None:
    """``!colab [arg]`` or ``!whoami`` (the whole message); else None."""
    stripped = (text or "").strip()
    if _WHOAMI.fullmatch(stripped):
        return Command("whoami")
    match = _COLAB.fullmatch(stripped)
    if match:
        return Command("colab", " ".join((match.group(1) or "").split()).lower())
    return None


async def run_command(
    command: Command,
    request: Any,
    *,
    colab: Any,
    admin_user_ids: frozenset[str],
    clock: Any,
) -> str:
    """The reply to *command* from the sender of *request*.

    ``colab`` is the :class:`~src.services.colab.endpoint.ColabEndpoint`,
    or None when COLAB_ENABLED is off; ``clock`` returns epoch seconds.
    """
    if command.name == "whoami":
        if request.source_type != "user":
            return WHOAMI_DM_ONLY
        return f"你的 LINE userId：{request.user_id}" if request.user_id else "讀不到你的 userId。"

    user_id = (request.user_id or "").strip()
    if not user_id or user_id not in admin_user_ids:
        logger.info(f"!colab refused for a non-admin ({user_id[:8]}…)")
        return ADMIN_ONLY
    if colab is None:
        return COLAB_OFF_CONFIG
    arg = command.arg or "status"
    if arg not in ("on", "off", "status"):
        return COLAB_USAGE

    now = clock()
    try:
        if arg == "on":
            state = await colab.store.update(lambda current: turn_on(current, now))
            logger.info(f"!colab on by an admin: {state.status}")
            return (
                f"已開啟 Colab（目前 {state.status}），排程會在一分鐘內處理。"
                "Colab 就緒前，訊息一律先由 API 模型回覆。"
            )
        if arg == "off":
            state = await colab.store.update(lambda current: turn_off(current, now))
            logger.info(f"!colab off by an admin: {state.status}")
            return f"已關閉 Colab（目前 {state.status}）；runtime 會由排程釋放，直到 !colab on 才會再開機。"
        state = await colab.store.load()
    except ColabStateError as exc:
        logger.warning(f"!colab {arg}: {exc}")
        return STATE_UNAVAILABLE
    return format_status(state, colab, now)


def turn_on(state: ColabState, now: float) -> ColabState:
    """``!colab on``: auto again, out of DISABLED/COOLDOWN, a manual start now."""
    changes: dict[str, Any] = {"desired": AUTO, "manual_demand_at": now, "hold_reason": ""}
    if state.status in (DISABLED, COOLDOWN):
        return state.with_status(
            IDLE,
            now,
            disabled_reason="",
            cooldown_until=None,
            cooldown_step=0,
            failures=0,
            **changes,
        )
    return replace(state, **changes)


def turn_off(state: ColabState, now: float) -> ColabState:
    """``!colab off``: stays down; a session being assigned, booting or ready drains now."""
    if state.status in (ASSIGNING, BOOTING, READY):
        return state.with_status(DRAINING, now, desired=OFF, last_error="!colab off")
    return replace(state, desired=OFF)


def format_status(state: ColabState, colab: Any, now: float) -> str:
    """``!colab status``: a few zh-TW lines (no URL, token or ids)."""
    lines = [f"Colab 狀態：{_STATUS_NAMES.get(state.status, state.status)}"]
    if state.since:
        lines[0] += f"，自 {_when(state.since)} 起"
    lines.append(
        f"開機方式：{'自動（有訊息就開機）' if colab.auto_start else '手動（!colab on）'}"
        + ("；目前已關閉（!colab off）" if state.desired == OFF else "")
    )
    if state.status == READY:
        lines.append(f"Tunnel：{'就緒' if colab.is_ready() else '健康檢查未通過，暫不使用'}")
        # The router's reason to sit this session out, one line per model:
        # what the readiness probe or a later health check found.
        reasons = {
            model: (
                f"沒有完全載入 GPU（VRAM {protocol.mib(in_vram)} / {protocol.mib(size)} MiB）"
                if size
                else "沒有載入（/api/ps 找不到）"
            )
            for model, (in_vram, size) in colab.off_gpu().items()
        }
        for model, reason in reasons.items():
            lines.append(f"{model}：{reason}，這次不用它路由（改用 API）")
        # The models follow their tags: a moved one is news, not a problem.
        for model, (verified, actual) in colab.moved_tags().items():
            lines.append(f"{model}：tag 已更新（digest {actual[:12]}，驗證過的是 {verified[:12]}）")
    if state.ccu_balance is not None:
        rate = f"，每小時 {state.ccu_rate:.2f}" if state.ccu_rate is not None else ""
        lines.append(f"CU 餘額：{state.ccu_balance:.1f}{rate}（{_when(state.ccu_checked_at)} 讀取）")
    lines.append(
        f"今日用量：{state.cu_used_today:.1f} / {colab.limits.daily_cu_cap:g} CU"
        f"（開機 {state.starts_today} 次）"
    )
    if state.hold_reason:
        lines.append(f"暫不開機：{state.hold_reason}")
    if state.cooldown_until and state.cooldown_until > now:
        lines.append(f"冷卻到：{_when(state.cooldown_until)}")
    if state.disabled_reason:
        lines.append(f"停用原因：{state.disabled_reason}")
    if state.last_error:
        lines.append(f"最後錯誤：{state.last_error}")
    return "\n".join(lines)


def _when(epoch: float | None) -> str:
    if not epoch:
        return "?"
    return datetime.fromtimestamp(epoch, _TIMEZONE).strftime("%m-%d %H:%M")
