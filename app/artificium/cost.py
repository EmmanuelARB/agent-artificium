"""What model requests cost: prices, the per-day ledger, and the daily limit.

Prices belong to a connection (its section in ``config.json``). A cost is
either reported by the provider in the response (it then wins) or computed
from the configured prices and the normalized token counts; without either it
is unknown, and the tokens are still counted. The ledger is one small JSON
file with a record per local day, so neither the request path nor the state
header ever scans a log. The limit and the 80% notice read the same ledger.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .filesystem import Paths, atomic_write_json, read_json

# Per million tokens. Reasoning tokens are part of the output count (see
# usage.normalize_usage) and are billed at the output price.
PRICE_FIELDS = ("price_input", "price_cached_input", "price_cache_write", "price_output")
# Adapters whose responses carry the cost the provider charged (usage.cost).
REPORTS_COST_ADAPTERS = frozenset({"openrouter"})
# The share of the daily limit at which the agent is told once per day.
WARN_FRACTION = 0.8
# Days of ledger kept; older days are dropped on write.
KEEP_DAYS = 60
TOKEN_KINDS = ("input_tokens", "output_tokens", "reasoning_tokens",
               "cache_read_tokens", "cache_write_tokens")
CONNECTIONS = ("worker", "oversight")


class CostLimitReached(RuntimeError):
    """Raised instead of sending a request once the daily limit is reached."""

    cost_limited = True

    def __init__(self, message: str, *, reset_epoch: float | None = None):
        super().__init__(message)
        self.reset_epoch = reset_epoch


# -- configuration helpers ----------------------------------------------------

def check_price(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or not 0 <= value <= 1_000_000:
        raise ValueError(f"{name} must be null or a price per million tokens from 0 to 1,000,000")


def check_currency(value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 12:
        raise ValueError("price_currency must be a short label such as USD")


def check_limit(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError("daily_cost_limit must be null (no limit) or a positive amount")


def check_timezone(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError("cost_day_timezone must be null (this machine's zone) or an IANA name")
    try:
        ZoneInfo(value.strip())
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise ValueError(
            f"cost_day_timezone {value!r} is not a known IANA time zone (for example Europe/Paris)"
        ) from None


def has_prices(config: Any) -> bool:
    """Input and output prices are the minimum for a computed cost."""
    return getattr(config, "price_input", None) is not None \
        and getattr(config, "price_output", None) is not None


def reports_cost(config: Any) -> bool:
    return getattr(config, "adapter", None) in REPORTS_COST_ADAPTERS


def limit_currency(config: Any) -> str:
    """The label shown with the limit: that of the first connection with prices."""
    separate = getattr(config, "oversight_connection", None)
    for source in (vars(config) if hasattr(config, "__dict__") else {},
                   separate if isinstance(separate, dict) else {}):
        if source.get("price_input") is not None or source.get("price_output") is not None:
            return str(source.get("price_currency") or "USD").strip()
    return str(getattr(config, "price_currency", None) or "USD").strip()


def cost_limit_problem(config: Any) -> str | None:
    """Why the daily limit of this configuration could not be enforced, or None.

    The limit covers the worker and the oversight connection together, so each
    needs prices (zero for a free local model) or a provider that reports
    costs, and the prices need one currency label.
    """
    if getattr(config, "daily_cost_limit", None) is None:
        return None
    covered: list[tuple[str, Any, str]] = [
        ("worker", config, "python3 artificium.py configure model")]
    build = getattr(config, "oversight_config", None)
    separate = build() if callable(build) else None
    if separate is not None:
        covered.append(("oversight", separate, "python3 artificium.py configure oversight"))
    elif getattr(config, "oversight_model", None) and config.oversight_model != config.model:
        covered.append(("oversight", None, ""))
    for label, connection, command in covered:
        if connection is None:
            return (
                f"The daily cost limit covers the oversight model {config.oversight_model!r} "
                "(set as oversight_model on the worker's endpoint), which has no prices of its own. "
                "Give oversight a connection with prices (python3 artificium.py configure oversight "
                "--provider ... --model ... --price-input ... --price-output ...) or clear oversight_model.")
        if has_prices(connection) or reports_cost(connection):
            continue
        return (
            f"The daily cost limit covers the {label} connection ({connection.provider}/{connection.model}), "
            "which has no prices and whose provider does not report costs. Set them, per million tokens: "
            f"{command} --price-input AMOUNT --price-output AMOUNT "
            "[--price-cached-input AMOUNT] [--price-cache-write AMOUNT] "
            "(use 0 for a free local model).")
    currencies = {str(c.price_currency).strip() for _, c, _ in covered
                  if c is not None and has_prices(c)}
    if len(currencies) > 1:
        return ("The daily cost limit adds both connections together without conversion, but "
                f"their prices use different currency labels ({', '.join(sorted(currencies))}). "
                "Use the same --price-currency for both.")
    return None


# -- pricing one response -------------------------------------------------------

def compute_cost(config: Any, usage: dict[str, Any]) -> float | None:
    """Cost from the configured prices and normalized usage, or None if unknown.

    ``input_tokens`` is the whole prompt; the cache-read and cache-write parts
    are priced separately (at the input price when no dedicated price is set,
    which never understates), the rest at the input price. Output includes
    reasoning tokens.
    """
    price_in = getattr(config, "price_input", None)
    price_out = getattr(config, "price_output", None)
    if price_in is None or price_out is None:
        return None
    total_in, out = usage.get("input_tokens"), usage.get("output_tokens")
    if total_in is None and out is None:
        return None
    total_in, out = total_in or 0, out or 0
    read = min(usage.get("cache_read_tokens") or 0, total_in)
    write = min(usage.get("cache_write_tokens") or 0, total_in - read)
    cached = getattr(config, "price_cached_input", None)
    written = getattr(config, "price_cache_write", None)
    micro = ((total_in - read - write) * price_in
             + read * (price_in if cached is None else cached)
             + write * (price_in if written is None else written)
             + out * price_out)
    return micro / 1_000_000


def reported_cost(config: Any, usage: dict[str, Any] | None) -> float | None:
    """The cost the provider says it charged, when its adapter reports one."""
    if not reports_cost(config) or not isinstance(usage, dict):
        return None
    value = usage.get("cost")
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def price_usage(config: Any, raw_usage: dict[str, Any] | None,
                normalized: dict[str, Any]) -> tuple[float | None, str]:
    """(cost, source) with source ``reported``, ``computed`` or ``unknown``."""
    reported = reported_cost(config, raw_usage)
    if reported is not None:
        return reported, "reported"
    computed = compute_cost(config, normalized)
    if computed is not None:
        return computed, "computed"
    return None, "unknown"


# -- the day -------------------------------------------------------------------

class Day:
    """Local calendar days in one zone (the machine's when none is named)."""

    def __init__(self, timezone: str | None):
        self.name = timezone.strip() if isinstance(timezone, str) and timezone.strip() else None
        self.zone = ZoneInfo(self.name) if self.name else None

    @property
    def label(self) -> str:
        return self.name or "local time"

    def _date(self, epoch: float) -> dt.date:
        if self.zone is None:
            return dt.date(*time.localtime(epoch)[:3])
        return dt.datetime.fromtimestamp(epoch, self.zone).date()

    def key(self, epoch: float) -> str:
        return self._date(epoch).isoformat()

    def next_reset(self, epoch: float) -> float:
        """The epoch of the next local midnight."""
        following = self._date(epoch) + dt.timedelta(days=1)
        if self.zone is None:
            return time.mktime((following.year, following.month, following.day, 0, 0, 0, 0, 0, -1))
        return dt.datetime.combine(following, dt.time(0), tzinfo=self.zone).timestamp()

    def clock_text(self, epoch: float) -> str:
        """``YYYY-MM-DD HH:MM`` in this zone."""
        if self.zone is None:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))
        return dt.datetime.fromtimestamp(epoch, self.zone).strftime("%Y-%m-%d %H:%M")


def _empty_connection() -> dict[str, Any]:
    entry: dict[str, Any] = {"requests": 0, "cost": 0.0, "unknown_cost_requests": 0,
                             "currency": "USD"}
    entry.update({kind: 0 for kind in TOKEN_KINDS})
    return entry


class CostLedger:
    """Per-day totals per connection, in ``logs/runtime/cost-ledger.json``.

    Every call re-reads the small file, so a second reader (the ``cost``
    command, a test) always sees what the life-loop wrote.
    """

    def __init__(self, paths: Paths, config: Any, clock: Callable[[], float] = time.time):
        self.paths = paths
        self.config = config
        self.clock = clock

    @property
    def path(self):
        return self.paths.runtime / "cost-ledger.json"

    @property
    def day(self) -> Day:
        return Day(getattr(self.config, "cost_day_timezone", None))

    def _load(self) -> dict[str, Any]:
        value = read_json(self.path, {})
        value = value if isinstance(value, dict) else {}
        if not isinstance(value.get("days"), dict):
            value["days"] = {}
        return value

    def record(self, connection: str, normalized: dict[str, Any], cost: float | None,
               *, currency: str = "USD") -> None:
        """Add one billed request to today's totals."""
        data = self._load()
        key = self.day.key(self.clock())
        entry = data["days"].setdefault(key, {"connections": {}, "warned": False})
        total = entry.setdefault("connections", {}).setdefault(connection, _empty_connection())
        total["requests"] = int(total.get("requests", 0)) + 1
        total["currency"] = currency
        for kind in TOKEN_KINDS:
            total[kind] = int(total.get(kind, 0)) + int(normalized.get(kind) or 0)
        if cost is None:
            total["unknown_cost_requests"] = int(total.get("unknown_cost_requests", 0)) + 1
        else:
            total["cost"] = round(float(total.get("cost", 0.0)) + cost, 8)
        for old in sorted(data["days"])[:-KEEP_DAYS]:
            del data["days"][old]
        atomic_write_json(self.path, data)

    def mark_warned(self) -> None:
        data = self._load()
        key = self.day.key(self.clock())
        data["days"].setdefault(key, {"connections": {}, "warned": False})["warned"] = True
        atomic_write_json(self.path, data)

    @staticmethod
    def _spent(entry: dict[str, Any]) -> float:
        return round(sum(float(item.get("cost", 0.0) or 0.0)
                         for item in (entry.get("connections") or {}).values()
                         if isinstance(item, dict)), 8)

    def status(self) -> dict[str, Any]:
        """Today's position against the limit; cheap enough for every request."""
        now = self.clock()
        day = self.day
        key = day.key(now)
        entry = self._load()["days"].get(key)
        entry = entry if isinstance(entry, dict) else {}
        connections = {name: item for name, item in (entry.get("connections") or {}).items()
                       if isinstance(item, dict)}
        spent = self._spent(entry)
        limit = getattr(self.config, "daily_cost_limit", None)
        reset = day.next_reset(now)
        known = sum(int(item.get("requests", 0)) - int(item.get("unknown_cost_requests", 0))
                    for item in connections.values())
        return {
            "day": key, "timezone": day.label, "spent": spent, "limit": limit,
            "currency": limit_currency(self.config),
            "fraction": spent / limit if limit else None,
            "limited": limit is not None and spent >= limit,
            "warn_due": (limit is not None and spent >= WARN_FRACTION * limit
                         and spent < limit and not entry.get("warned")),
            "reset_epoch": reset, "reset_at": day.clock_text(reset),
            "requests": sum(int(item.get("requests", 0)) for item in connections.values()),
            "unknown_cost_requests": sum(int(item.get("unknown_cost_requests", 0))
                                         for item in connections.values()),
            "known_cost_requests": known,
            "connections": connections,
        }

    def report(self, days: int = 7) -> dict[str, Any]:
        """The last ``days`` days that have records, newest first, plus today."""
        data = self._load()["days"]
        shown = []
        for key in sorted(data, reverse=True)[:max(1, days)]:
            entry = data[key]
            if isinstance(entry, dict):
                shown.append({"day": key, "spent": self._spent(entry),
                              "connections": entry.get("connections") or {}})
        return {"today": self.status(), "days": shown}


# -- what the agent and the operator read ---------------------------------------

def money(amount: float) -> str:
    return f"{amount:,.2f}" if amount >= 0.01 or amount == 0 else f"{amount:.4f}"


def header_line(config: Any, status: dict[str, Any], priced: bool) -> str:
    """One short line for the state header, or "" when there is nothing to show."""
    if status["limit"] is None and not priced and not status["known_cost_requests"]:
        return ""
    currency = status["currency"]
    text = f"Spend today: {money(status['spent'])}"
    if status["limit"] is not None:
        text += (f" of {money(status['limit'])} {currency} limit "
                 f"({status['fraction'] * 100:.0f}%)")
    else:
        text += f" {currency}"
    if status["unknown_cost_requests"]:
        text += f"; {status['unknown_cost_requests']} requests of unknown cost not counted"
    return text


def format_report(report: dict[str, Any]) -> str:
    today = report["today"]
    currency = today["currency"]
    lines = [f"Cost, day boundary {today['timezone']}; today is {today['day']}"]
    if today["limit"] is None:
        lines.append("Daily limit: none")
    else:
        lines.append(
            f"Daily limit: {money(today['limit'])} {currency} for both connections together; "
            f"spent {money(today['spent'])} ({today['fraction'] * 100:.0f}%)"
            + ("; REACHED, requests are held back" if today["limited"] else ""))
        lines.append(f"Resets at {today['reset_at']} ({today['timezone']})")
    for entry in report["days"]:
        label = "today" if entry["day"] == today["day"] else entry["day"]
        lines.append(f"{label}: {money(entry['spent'])}")
        for name in CONNECTIONS:
            item = entry["connections"].get(name)
            if not isinstance(item, dict):
                continue
            unknown = int(item.get("unknown_cost_requests", 0))
            lines.append(
                f"  {name}: {item.get('requests', 0)} requests; "
                f"{int(item.get('input_tokens', 0)):,} in "
                f"({int(item.get('cache_read_tokens', 0)):,} cached read, "
                f"{int(item.get('cache_write_tokens', 0)):,} cache write), "
                f"{int(item.get('output_tokens', 0)):,} out "
                f"({int(item.get('reasoning_tokens', 0)):,} reasoning); "
                f"{money(float(item.get('cost', 0.0)))} {item.get('currency', currency)}"
                + (f"; {unknown} requests of unknown cost" if unknown else ""))
    if not report["days"]:
        lines.append("No model requests recorded yet.")
    lines.append("Computed costs are estimates from the configured prices; "
                 "a cost the provider reports replaces the estimate.")
    return "\n".join(lines)
