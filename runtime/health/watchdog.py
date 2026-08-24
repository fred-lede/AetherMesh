from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from runtime.alerting.alert_manager import AlertManager, get_alert_manager
from runtime.alerting.notifier_base import Severity

logger = logging.getLogger("health.watchdog")

_PORT_ENV_MAP = {
    "control_plane": ("AIIH_CONTROL_PORT", 9200),
    "openai_router": ("AIIH_ROUTER_PORT", 8001),
    "anthropic_router": ("AIIH_ANTHROPIC_PORT", 8002),
    "dashboard": ("AIIH_DASHBOARD_PORT", 9001),
    "metrics": ("AIIH_METRICS_PORT", 9100),
    "node_agent": ("AIIH_NODE_PORT", 9400),
    "worker_agent": ("AIIH_WORKER_PORT", 9300),
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "interval_s": 30,
    "health_timeout_s": 10,
    "hang_failures_to_alert": 3,
    "startup_grace_s": 180,
    "exclude_services": [],
    "rules": {
        "service_rss_mb": {"warn": 4096, "critical": 8192},
        "disk_free_pct": {"warn": 15.0, "critical": 5.0},
    },
    "auto_restart": {
        "enabled": False,
        "restart_after_s": 60,
        "cooldown_s": 1800,
        "max_per_day": 4,
        "exclude": [],
    },
}


def merged_watchdog_config(section: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**_DEFAULT_CONFIG}
    if not isinstance(section, dict):
        return merged
    for key, value in section.items():
        if key == "rules" and isinstance(value, dict):
            rules = {**_DEFAULT_CONFIG["rules"]}
            for rule_name, rule_val in value.items():
                if isinstance(rule_val, dict):
                    rules[rule_name] = {**rules.get(rule_name, {}), **rule_val}
                else:
                    rules[rule_name] = rule_val
            merged["rules"] = rules
        elif key == "auto_restart" and isinstance(value, dict):
            merged["auto_restart"] = {**_DEFAULT_CONFIG["auto_restart"], **value}
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class ServiceCheckResult:
    name: str
    alive: bool = True
    healthy: bool | None = None
    rss_mb: float | None = None
    detail: str = ""
    in_grace: bool = False
    startup_age_s: float | None = None


@dataclass(slots=True)
class _ServiceState:
    hang_since: float | None = None
    dead_since: float | None = None
    consecutive_hang: int = 0
    restart_count_today: int = 0
    restart_day: str = ""
    last_restart_ts: float = 0.0


class Watchdog:
    def __init__(
        self,
        get_services: Callable[[], dict[str, Any]],
        alert_manager: AlertManager | None = None,
        restart_fn: Callable[[str], bool] | None = None,
        config_section: dict[str, Any] | None = None,
    ) -> None:
        self._get_services = get_services
        self._alerts = alert_manager or get_alert_manager()
        self._restart_fn = restart_fn
        if config_section is not None:
            self._config_override = lambda: config_section
            self._config_path: str | None = None
        else:
            from config.settings import settings

            self._config_path = str(settings.config_path("notifications.json"))
            self._config_override = lambda: (self._alerts.config.get("watchdog", {}) if hasattr(self._alerts, "config") else {})
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._states: dict[str, _ServiceState] = {}
        self.last_summary: dict[str, Any] = {}

    def load_config(self) -> dict[str, Any]:
        if self._config_path:
            import json

            try:
                data = json.loads(Path(self._config_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            return merged_watchdog_config(data.get("watchdog") if isinstance(data, dict) else None)
        return merged_watchdog_config(self._config_override())

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        cfg = self.load_config()
        if not cfg.get("enabled", True):
            logger.info("watchdog disabled by config")
            return
        interval = max(5, int(cfg.get("interval_s", 30)))
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.wait(interval):
                try:
                    self.check_once()
                except Exception:
                    logger.exception("watchdog check failed")

        self._thread = threading.Thread(target=_loop, daemon=True, name="aether-watchdog")
        self._thread.start()
        logger.info("watchdog started (interval=%ss)", interval)

    def stop(self) -> None:
        self._stop_event.set()

    def check_once(self) -> dict[str, Any]:
        try:
            self._alerts.reload_if_changed()
        except Exception:
            pass
        cfg = self.load_config()
        results = self._check_services(cfg)
        disk_pct = self._check_disk(cfg)
        self.last_summary = {
            "timestamp": time.time(),
            "services": {
                r.name: {
                    "alive": r.alive,
                    "healthy": r.healthy,
                    "rss_mb": r.rss_mb,
                    "in_grace": r.in_grace,
                    "startup_age_s": r.startup_age_s,
                }
                for r in results
            },
            "disk_free_pct": disk_pct,
        }
        return self.last_summary

    @staticmethod
    def _port_for(name: str) -> int | None:
        import os

        entry = _PORT_ENV_MAP.get(name)
        if entry is None:
            return None
        env_key, default = entry
        try:
            return int(os.getenv(env_key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _rss_mb(pid: int) -> float | None:
        try:
            import psutil

            return round(psutil.Process(pid).memory_info().rss / (1024 * 1024), 1)
        except Exception:
            return None

    @staticmethod
    def _process_age_s(pid: int, now: float) -> float | None:
        try:
            import psutil

            return max(0.0, now - psutil.Process(pid).create_time())
        except Exception:
            return None

    def _check_services(self, cfg: dict[str, Any]) -> list[ServiceCheckResult]:
        timeout = float(cfg.get("health_timeout_s", 10))
        grace_s = float(cfg.get("startup_grace_s", 180))
        excluded = set(cfg.get("exclude_services") or [])
        services = self._get_services() or {}
        now = time.time()
        results: list[ServiceCheckResult] = []
        for name, sp in services.items():
            if name in excluded:
                continue
            if getattr(sp, "intentionally_stopped", False):
                self._states.pop(name, None)
                continue
            result = ServiceCheckResult(name=name)
            state = self._states.setdefault(name, _ServiceState())

            status = getattr(sp, "status", "unknown")
            result.alive = status == "running"

            proc = getattr(sp, "process", None)
            pid = getattr(proc, "pid", None)
            if result.alive and proc is not None and pid:
                poll = proc.poll()
                result.alive = poll is None
                if result.alive:
                    result.rss_mb = self._rss_mb(pid)
                    age = self._process_age_s(pid, now)
                    if age is not None:
                        result.startup_age_s = round(age, 1)
                        result.in_grace = age < grace_s

            if result.alive:
                port = self._port_for(name)
                if port is not None:
                    try:
                        resp = requests.get(
                            f"http://127.0.0.1:{port}/health", timeout=timeout
                        )
                        result.healthy = resp.status_code < 500
                    except requests.RequestException:
                        result.healthy = False
                        result.detail = f"/health 無回應（timeout {timeout}s）"

            self._evaluate_service(cfg, state, result, now)
            results.append(result)
        return results

    def _check_disk(self, cfg: dict[str, Any]) -> float | None:
        try:
            usage = shutil.disk_usage(Path.cwd())
        except OSError:
            return None
        free_pct = round(usage.free / usage.total * 100, 1) if usage.total else None
        rule = cfg.get("rules", {}).get("disk_free_pct", {})
        if free_pct is None or not isinstance(rule, dict):
            return free_pct
        critical = rule.get("critical")
        warn = rule.get("warn")
        if critical is not None and free_pct <= float(critical):
            self._alerts.dispatch(
                title="磁碟空間嚴重不足",
                message=f"剩餘空間僅 {free_pct}%（低於 critical 門檻 {critical}%）",
                severity=Severity.CRITICAL,
                rule_key="disk_critical",
                cooldown_s=3600,
            )
        elif warn is not None and free_pct <= float(warn):
            self._alerts.dispatch(
                title="磁碟空間不足",
                message=f"剩餘空間 {free_pct}%（低於 warn 門檻 {warn}%）",
                severity=Severity.WARNING,
                rule_key="disk_warn",
                cooldown_s=3600,
            )
        return free_pct

    def _evaluate_service(
        self,
        cfg: dict[str, Any],
        state: _ServiceState,
        result: ServiceCheckResult,
        now: float,
    ) -> None:
        hang_alert_after = int(cfg.get("hang_failures_to_alert", 3))
        rss_rule = cfg.get("rules", {}).get("service_rss_mb", {})

        if result.in_grace:
            state.dead_since = None
            state.consecutive_hang = 0
            state.hang_since = None
            self._rss_alerts(result, rss_rule)
            return

        if not result.alive:
            if state.dead_since is None:
                state.dead_since = now
            dead_s = int(now - state.dead_since)
            self._alerts.dispatch(
                title=f"服務停止運作：{result.name}",
                message=f"{result.name} process 已停止 {dead_s}s",
                severity=Severity.CRITICAL,
                rule_key=f"dead:{result.name}",
                cooldown_s=600,
            )
            self._maybe_autorestart(cfg, state, result.name, now, reason=f"process 停止 {dead_s}s")
            state.consecutive_hang = 0
            state.hang_since = None
            return

        state.dead_since = None

        if result.healthy is False:
            state.consecutive_hang += 1
            if state.hang_since is None:
                state.hang_since = now
            hang_s = int(now - state.hang_since)
            if state.consecutive_hang >= hang_alert_after:
                self._alerts.dispatch(
                    title=f"服務無回應：{result.name}",
                    message=(
                        f"{result.name} 連續 {state.consecutive_hang} 次 /health 失敗"
                        f"（持續 {hang_s}s）。{result.detail}"
                    ),
                    severity=Severity.WARNING if hang_s < 300 else Severity.CRITICAL,
                    rule_key=f"hang:{result.name}",
                    cooldown_s=600,
                )
                self._maybe_autorestart(
                    cfg, state, result.name, now, reason=f"無回應 {hang_s}s"
                )
        else:
            if state.consecutive_hang >= hang_alert_after:
                self._alerts.dispatch(
                    title=f"服務恢復：{result.name}",
                    message=f"{result.name} /health 通過（先前連續失敗 {state.consecutive_hang} 次）",
                    severity=Severity.INFO,
                    rule_key=f"recovered:{result.name}",
                    cooldown_s=60,
                )
            state.consecutive_hang = 0
            state.hang_since = None

        self._rss_alerts(result, rss_rule)

    def _rss_alerts(self, result: ServiceCheckResult, rss_rule: dict[str, Any]) -> None:
        if result.rss_mb is not None and isinstance(rss_rule, dict) and rss_rule.get("warn"):
            warn_mb = float(rss_rule["warn"])
            critical_mb = rss_rule.get("critical")
            if critical_mb and result.rss_mb >= float(critical_mb):
                self._alerts.dispatch(
                    title=f"記憶體過高：{result.name}",
                    message=(
                        f"{result.name} RSS {result.rss_mb} MB ≥ critical "
                        f"{critical_mb} MB，疑似洩漏"
                    ),
                    severity=Severity.CRITICAL,
                    rule_key=f"rss_crit:{result.name}",
                    cooldown_s=1800,
                )
            elif result.rss_mb >= warn_mb:
                self._alerts.dispatch(
                    title=f"記憶體偏高：{result.name}",
                    message=f"{result.name} RSS {result.rss_mb} MB ≥ warn {warn_mb} MB",
                    severity=Severity.WARNING,
                    rule_key=f"rss_warn:{result.name}",
                    cooldown_s=1800,
                )

    def _maybe_autorestart(
        self,
        cfg: dict[str, Any],
        state: _ServiceState,
        name: str,
        now: float,
        reason: str,
    ) -> bool:
        ar = cfg.get("auto_restart", {})
        if not isinstance(ar, dict) or not ar.get("enabled", False):
            return False
        if name in set(ar.get("exclude", [])):
            return False
        if self._restart_fn is None:
            logger.warning("auto_restart enabled but no restart_fn provided")
            return False

        since = state.dead_since if state.dead_since is not None else state.hang_since
        if since is None or now - since < float(ar.get("restart_after_s", 60)):
            return False

        today = time.strftime("%Y-%m-%d")
        if state.restart_day != today:
            state.restart_day = today
            state.restart_count_today = 0
        max_per_day = int(ar.get("max_per_day", 4))
        if state.restart_count_today >= max_per_day:
            self._alerts.dispatch(
                title=f"自動重啟已達上限：{name}",
                message=f"{name} 今日已自動重啟 {state.restart_count_today}/{max_per_day} 次，跳過（原因：{reason}）",
                severity=Severity.CRITICAL,
                rule_key=f"restart_cap:{name}",
                cooldown_s=3600,
            )
            return False
        if now - state.last_restart_ts < float(ar.get("cooldown_s", 1800)):
            return False

        state.restart_count_today += 1
        state.last_restart_ts = now
        ok = False
        try:
            ok = bool(self._restart_fn(name))
        except Exception as exc:
            logger.exception("auto_restart failed for %s: %s", name, exc)
        self._alerts.dispatch(
            title=f"自動重啟：{name}",
            message=(
                f"原因：{reason}\n結果：{'成功' if ok else '失敗'}\n"
                f"今日第 {state.restart_count_today}/{max_per_day} 次"
            ),
            severity=Severity.INFO if ok else Severity.CRITICAL,
            rule_key=f"restart:{name}",
            cooldown_s=60,
        )
        if ok:
            state.dead_since = None
            state.hang_since = None
            state.consecutive_hang = 0
        return ok
