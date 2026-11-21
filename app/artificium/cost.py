"""What model requests cost: prices, the per-day ledger, and the spend limits.

Prices belong to a connection (its section in ``config.json``). A cost is
either reported by the provider in the response (it then wins) or computed
from the configured prices and the normalized token counts; without either it
is unknown, and the tokens are still counted. The ledger is one small JSON
file with a record per local day, so neither the request path nor the state
header ever scans a log. The limits and the 80% notices read the same ledger.

A limit can cover four periods (see LIMIT_PERIODS), each counted over both
connections together. The day, week and month are calendar periods derived
from the per-day records; the five-hour period is a rolling window, so the
ledger also keeps the cost and time of the requests of the last five hours.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from typing import Any, Callable, NamedTuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .filesystem import Paths, atomic_write_json, read_json

# Per million tokens. Reasoning tokens are part of the output count (see
# usage.normalize_usage) and are billed at the output price.
PRICE_FIELDS = ("price_input", "price_cached_input", "price_cache_write", "price_output")
# Adapters whose responses carry the cost the provider charged (usage.cost).
REPORTS_COST_ADAPTERS = frozenset({"openrouter", "opencode_go"})
# Adapters whose responses also say how much of a plan's allowance a request
# used (usage.opencode.allowance, in dollars of the plan's monthly limit) and
# what it would cost at list price. The spend limits never count these: they
# counts dollars actually charged (usage.cost, 0 within a plan's limits).
PLAN_USAGE_FIELDS = ("allowance", "list_cost")
REPORTS_PLAN_USAGE_ADAPTERS = frozenset({"opencode_go"})
# The share of a limit at which the agent is told once per period.
WARN_FRACTION = 0.8
# Days of ledger kept; older days are dropped on write.
KEEP_DAYS = 60
TOKEN_KINDS = ("input_tokens", "output_tokens", "reasoning_tokens",
               "cache_read_tokens", "cache_write_tokens")
CONNECTIONS = ("worker", "oversight")


class Period(NamedTuple):
    key: str
    field: str       # the harness setting that holds this period's limit
    short: str       # in the state header and the status line
    phrase: str      # in a sentence: "Spent {phrase}"
    adjective: str   # "{adjective} cost limit"


ROLLING_HOURS = 5
# Shown in this order. "5h" is a rolling window of the last ROLLING_HOURS
# hours; the others are calendar periods in cost_day_timezone (the week starts
# on Monday), so the day and the month end at local midnight.
LIMIT_PERIODS = (
    Period("5h", "five_hour_cost_limit", "5h", "in the last 5 hours", "5-hour"),
    Period("day", "daily_cost_limit", "today", "today", "daily"),
    Period("week", "weekly_cost_limit", "week", "this week", "weekly"),
    Period("month", "monthly_cost_limit", "month", "this month", "monthly"),
)
LIMIT_FIELDS = tuple(item.field for item in LIMIT_PERIODS)


def period_by_key(key: str) -> Period:
    return next(item for item in LIMIT_PERIODS if item.key == key)


class CostLimitReached(RuntimeError):
    """Raised instead of sending a request once a spend limit is reached."""

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


def check_limit(value: Any, name: str = "daily_cost_limit") -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be null (no limit) or a positive amount")


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


def configured_limits(config: Any) -> dict[str, float | None]:
    """The limit of each period, None where there is none."""
    return {item.key: getattr(config, item.field, None) for item in LIMIT_PERIODS}


def limits_configured(config: Any) -> bool:
    return any(value is not None for value in configured_limits(config).values())


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
    """Why the spend limits of this configuration could not be enforced, or None.

    A limit covers the worker and the oversight connection together, so each
    needs prices (zero for a free local model) or a provider that reports
    costs, and the prices need one currency label.
    """
    if not limits_configured(config):
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
                f"The cost limits cover the oversight model {config.oversight_model!r} "
                "(set as oversight_model on the worker's endpoint), which has no prices of its own. "
                "Give oversight a connection with prices (python3 artificium.py configure oversight "
                "--provider ... --model ... --price-input ... --price-output ...) or clear oversight_model.")
        if has_prices(connection) or reports_cost(connection):
            continue
        return (
            f"The cost limits cover the {label} connection ({connection.provider}/{connection.model}), "
            "which has no prices and whose provider does not report costs. Set them, per million tokens: "
            f"{command} --price-input AMOUNT --price-output AMOUNT "
            "[--price-cached-input AMOUNT] [--price-cache-write AMOUNT] "
            "(use 0 for a free local model).")
    currencies = {str(c.price_currency).strip() for _, c, _ in covered
                  if c is not None and has_prices(c)}
    if len(currencies) > 1:
        return ("The cost limits add both connections together without conversion, but "
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


def reported_plan_usage(config: Any, usage: dict[str, Any] | None) -> dict[str, Any]:
    """Plan allowance and list-price cost of one response, when it reports them.

    Read generically from ``usage.opencode``: only finite non-negative numbers
    are kept (``allowance``, ``list_cost``), plus the ``billing_source`` label.
    Nothing is assumed about the plan's limits; the figures are shown as given.
    """
    if getattr(config, "adapter", None) not in REPORTS_PLAN_USAGE_ADAPTERS \
            or not isinstance(usage, dict) or not isinstance(usage.get("opencode"), dict):
        return {}
    detail = usage["opencode"]
    found: dict[str, Any] = {}
    for name in PLAN_USAGE_FIELDS:
        value = detail.get(name)
        if not isinstance(value, bool) and isinstance(value, (int, float)) \
                and math.isfinite(value) and value >= 0:
            found[name] = float(value)
    source = detail.get("billing_source")
    if found and isinstance(source, str) and source.strip():
        found["billing_source"] = source.strip()[:40]
    return found


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
    """Local calendar periods in one zone (the machine's when none is named)."""

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

    def _midnight(self, date: dt.date) -> float:
        if self.zone is None:
            return time.mktime((date.year, date.month, date.day, 0, 0, 0, 0, 0, -1))
        return dt.datetime.combine(date, dt.time(0), tzinfo=self.zone).timestamp()

    def next_reset(self, epoch: float) -> float:
        """The epoch of the next local midnight."""
        return self._midnight(self._date(epoch) + dt.timedelta(days=1))

    def period_start(self, period: str, epoch: float) -> dt.date:
        """First local day of the calendar period (day, week or month) of ``epoch``."""
        date = self._date(epoch)
        if period == "week":
            return date - dt.timedelta(days=date.weekday())
        if period == "month":
            return date.replace(day=1)
        return date

    def period_key(self, period: str, epoch: float) -> str:
        """Names one occurrence of a period: the month is ``YYYY-MM``, the others
        their first day."""
        start = self.period_start(period, epoch)
        return start.strftime("%Y-%m") if period == "month" else start.isoformat()

    def period_reset(self, period: str, epoch: float) -> float:
        """The epoch at which the calendar period of ``epoch`` ends (local midnight)."""
        start = self.period_start(period, epoch)
        if period == "week":
            following = start + dt.timedelta(days=7)
        elif period == "month":
            following = (start + dt.timedelta(days=32)).replace(day=1)
        else:
            following = start + dt.timedelta(days=1)
        return self._midnight(following)

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

    @staticmethod
    def _recent(data: dict[str, Any], now: float) -> list[tuple[float, float]]:
        """(time, cost) of the requests still inside the rolling window, oldest first."""
        found = []
        for item in data.get("recent") or []:
            try:
                at, amount = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if now - ROLLING_HOURS * 3600 < at and amount > 0:
                found.append((at, amount))
        return sorted(found)

    def record(self, connection: str, normalized: dict[str, Any], cost: float | None,
               *, currency: str = "USD", plan_usage: dict[str, Any] | None = None) -> None:
        """Add one billed request to today's totals.

        ``plan_usage`` (see :func:`reported_plan_usage`) adds the allowance and
        list-price sums, kept per connection and day beside the charged cost.
        """
        data = self._load()
        now = self.clock()
        key = self.day.key(now)
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
        for name in PLAN_USAGE_FIELDS:
            if plan_usage and name in plan_usage:
                total[name] = round(float(total.get(name, 0.0)) + plan_usage[name], 8)
        if plan_usage:
            total["plan_usage_requests"] = int(total.get("plan_usage_requests", 0)) + 1
        for old in sorted(data["days"])[:-KEEP_DAYS]:
            del data["days"][old]
        # Kept whether or not a five-hour limit is set, so turning one on
        # counts what was spent just before.
        recent = self._recent(data, now)
        if cost is not None and cost > 0:
            recent.append((now, float(cost)))
        data["recent"] = [[round(at, 3), round(amount, 8)] for at, amount in recent]
        atomic_write_json(self.path, data)

    def mark_warned(self, periods: list[str] | None = None) -> None:
        """Remember that the 80% notice was given, for these periods (default:
        every period that is due one)."""
        names = periods if periods is not None else [
            item["period"] for item in self.status()["periods"] if item["warn_due"]]
        data = self._load()
        now = self.clock()
        day = self.day
        warned = data.setdefault("warned_periods", {})
        for name in names:
            if name == "day":
                data["days"].setdefault(day.key(now), {"connections": {}, "warned": False})["warned"] = True
            elif name == "5h":
                warned[name] = now
            else:
                warned[name] = day.period_key(name, now)
        atomic_write_json(self.path, data)

    def note_limits(self, limits: dict[str, float | None]) -> dict[str, float | None] | None:
        """Store the limits in force now; returns those of the previous start, if any."""
        data = self._load()
        previous = data.get("limits_seen")
        data["limits_seen"] = limits
        atomic_write_json(self.path, data)
        return previous if isinstance(previous, dict) else None

    @staticmethod
    def _spent(entry: dict[str, Any]) -> float:
        return round(sum(float(item.get("cost", 0.0) or 0.0)
                         for item in (entry.get("connections") or {}).values()
                         if isinstance(item, dict)), 8)

    def _calendar_spent(self, data: dict[str, Any], start: dt.date, today: dt.date) -> float:
        first, last = start.isoformat(), today.isoformat()
        return round(sum(self._spent(entry) for key, entry in data["days"].items()
                         if isinstance(entry, dict) and first <= key <= last), 8)

    @staticmethod
    def _rolling_reset(events: list[tuple[float, float]], spent: float, limit: float) -> float:
        """When enough of the oldest spend has left the window to drop below the limit."""
        remaining = spent
        for at, amount in events:
            remaining = round(remaining - amount, 8)
            if remaining < limit:
                return at + ROLLING_HOURS * 3600 + 1
        return (events[-1][0] if events else 0.0) + ROLLING_HOURS * 3600 + 1

    def _period_entry(self, item: Period, data: dict[str, Any], now: float,
                      today_entry: dict[str, Any]) -> dict[str, Any]:
        day = self.day
        limit = getattr(self.config, item.field, None)
        rolling = item.key == "5h"
        if rolling:
            events = self._recent(data, now)
            spent = round(sum(amount for _, amount in events), 8)
        else:
            events = []
            spent = self._calendar_spent(data, day.period_start(item.key, now), day._date(now))
        limited = limit is not None and spent >= limit
        warned_periods = data.get("warned_periods") if isinstance(data.get("warned_periods"), dict) else {}
        if item.key == "day":
            warned = bool(today_entry.get("warned"))
        elif rolling:
            at = warned_periods.get("5h")
            warned = isinstance(at, (int, float)) and at > now - ROLLING_HOURS * 3600
        else:
            warned = warned_periods.get(item.key) == day.period_key(item.key, now)
        if rolling:
            if limited:
                reset = self._rolling_reset(events, spent, limit)
            else:
                reset = events[0][0] + ROLLING_HOURS * 3600 + 1 if events else None
            if reset is not None:
                # A time shown to the minute must never come before the moment.
                reset = math.ceil(reset / 60) * 60
        else:
            reset = day.period_reset(item.key, now)
        return {
            "period": item.key, "short": item.short, "phrase": item.phrase,
            "adjective": item.adjective, "rolling": rolling,
            "spent": spent, "limit": limit,
            "fraction": spent / limit if limit else None,
            "limited": limited,
            "warn_due": (limit is not None and spent >= WARN_FRACTION * limit
                         and spent < limit and not warned),
            "reset_epoch": reset, "reset_at": day.clock_text(reset) if reset is not None else "",
        }

    def status(self) -> dict[str, Any]:
        """The position against every limit; cheap enough for every request.

        The top-level ``spent``, ``limit``, ``fraction``, ``period`` and reset
        describe the period that decides the next pause: the limited period
        that frees up last, or else the one nearest its limit, or today when
        no limit is set. ``limited`` and ``warn_due`` hold when any period is.
        The day's own figures (``day``, ``connections``, requests) are kept as
        they always were.
        """
        now = self.clock()
        day = self.day
        key = day.key(now)
        data = self._load()
        entry = data["days"].get(key)
        entry = entry if isinstance(entry, dict) else {}
        connections = {name: item for name, item in (entry.get("connections") or {}).items()
                       if isinstance(item, dict)}
        periods = [self._period_entry(item, data, now, entry) for item in LIMIT_PERIODS]
        configured = [item for item in periods if item["limit"] is not None]
        limited = [item for item in configured if item["limited"]]
        if limited:
            binding = max(limited, key=lambda item: item["reset_epoch"])
        elif configured:
            binding = max(configured, key=lambda item: item["fraction"])
        else:
            binding = next(item for item in periods if item["period"] == "day")
        today_reset = day.next_reset(now)
        reset = binding["reset_epoch"] if binding["reset_epoch"] is not None else today_reset
        known = sum(int(item.get("requests", 0)) - int(item.get("unknown_cost_requests", 0))
                    for item in connections.values())
        return {
            "day": key, "timezone": day.label,
            "spent": binding["spent"], "limit": binding["limit"],
            "currency": limit_currency(self.config),
            "fraction": binding["fraction"],
            "period": binding["period"], "adjective": binding["adjective"],
            "phrase": binding["phrase"],
            "limited": bool(limited),
            "limited_periods": [item["period"] for item in limited],
            "warn_due": any(item["warn_due"] for item in periods),
            "periods": periods,
            "reset_epoch": reset, "reset_at": day.clock_text(reset),
            "spent_today": self._spent(entry),
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
    limited = [item for item in status.get("periods") or [] if item["limit"] is not None]
    if status["limit"] is None and not priced and not status["known_cost_requests"]:
        return ""
    currency = status["currency"]
    if len(limited) > 1 or (limited and limited[0]["period"] != "day"):
        parts = [f"{item['short']} {money(item['spent'])} of {money(item['limit'])} "
                 f"({item['fraction'] * 100:.0f}%)" for item in limited]
        text = f"Spend vs limits ({currency}): " + "; ".join(parts)
    else:
        text = f"Spend today: {money(status['spent'])}"
        if status["limit"] is not None:
            text += (f" of {money(status['limit'])} {currency} limit "
                     f"({status['fraction'] * 100:.0f}%)")
        else:
            text += f" {currency}"
    if status["unknown_cost_requests"]:
        text += f"; {status['unknown_cost_requests']} requests of unknown cost not counted"
    return text


def until_text(period: dict[str, Any], timezone: str) -> str:
    """When requests resume (or would), as a phrase for a message."""
    if period["rolling"]:
        return (f"until enough of the last 5 hours' spend has aged out "
                f"(from {period['reset_at']} ({timezone}) on)")
    return f"until {period['reset_at']} ({timezone})"


def limit_reached_text(status: dict[str, Any]) -> str:
    """Which limit holds requests back, for the error and the console."""
    limited = [item for item in status["periods"] if item["limited"]]
    binding = next(item for item in limited if item["period"] == status["period"]) if limited else None
    if binding is None:
        return "no cost limit is reached"
    text = (f"{binding['adjective']} cost limit reached ({money(binding['spent'])} of "
            f"{money(binding['limit'])} {status['currency']})")
    others = [item["adjective"] for item in limited if item is not binding]
    if others:
        text += "; also at the limit: " + ", ".join(others)
    return text


def _plan_usage_text(item: dict[str, Any]) -> str:
    if not int(item.get("plan_usage_requests", 0) or 0):
        return ""
    parts = []
    if "allowance" in item:
        parts.append(f"plan allowance used {float(item['allowance']):.4f}")
    if "list_cost" in item:
        parts.append(f"{float(item['list_cost']):.4f} at list price")
    return "; " + ", ".join(parts) if parts else ""


def format_report(report: dict[str, Any]) -> str:
    today = report["today"]
    currency = today["currency"]
    lines = [f"Cost, day boundary {today['timezone']}; today is {today['day']}"]
    configured = [item for item in today["periods"] if item["limit"] is not None]
    if not configured:
        lines.append("Daily limit: none (limits: five_hour_cost_limit, daily_cost_limit, "
                     "weekly_cost_limit, monthly_cost_limit)")
    for item in today["periods"]:
        name = item["adjective"].capitalize()
        if item["limit"] is None:
            if item["period"] != "day" and item["spent"]:
                lines.append(f"{name} limit: none; spent {item['phrase']}: {money(item['spent'])} {currency}")
            continue
        lines.append(
            f"{name} limit: {money(item['limit'])} {currency} for both connections together; "
            f"spent {money(item['spent'])} ({item['fraction'] * 100:.0f}%)"
            + ("; REACHED, requests are held back" if item["limited"] else ""))
        if item["rolling"]:
            if item["reset_epoch"] is not None:
                lines.append(("Requests resume at " if item["limited"]
                              else "The oldest spend leaves the window at ")
                             + f"{item['reset_at']} ({today['timezone']}); the window is the last "
                             f"{ROLLING_HOURS} hours")
        else:
            lines.append(f"Resets at {item['reset_at']} ({today['timezone']})")
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
                + (f"; {unknown} requests of unknown cost" if unknown else "")
                + _plan_usage_text(item))
    if not report["days"]:
        lines.append("No model requests recorded yet.")
    if any(isinstance(item, dict) and item.get("plan_usage_requests")
           for entry in report["days"] for item in entry["connections"].values()):
        lines.append("Plan allowance is in dollars of the plan's monthly limit; within the plan's "
                     "limits the charged cost is 0 and the allowance is the figure to watch. "
                     "The cost limits count charged cost only.")
    lines.append("Computed costs are estimates from the configured prices; "
                 "a cost the provider reports replaces the estimate.")
    return "\n".join(lines)
