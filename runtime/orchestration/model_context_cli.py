from __future__ import annotations

import argparse
import sys
from typing import Any

from config.settings import settings
from runtime.orchestration.model_context import (
    _base_url_for,
    fetch_context_length,
    refresh_auto_context,
)


def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="model-context",
        description="Resolve / auto-fetch model context length (token window).",
    )
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch", help="Fetch context length for a single model.")
    p_fetch.add_argument("model")
    p_fetch.add_argument("--provider", default="ollama")
    p_fetch.add_argument("--base-url")
    p_fetch.add_argument("--api-key")
    p_fetch.add_argument("--timeout", type=float, default=10.0)

    p_refresh = sub.add_parser(
        "refresh", help="Auto-fetch context length for models lacking one and write back."
    )
    p_refresh.add_argument("--write", action="store_true", help="Persist values into models.yaml")

    args = parser.parse_args()

    if args.command == "fetch":
        base_url = args.base_url
        if not base_url:
            entry = _entry_for(args.model)
            base_url = _base_url_for(entry, args.provider) if entry else None
        if not base_url:
            print(f"ERROR: no base_url resolved for '{args.model}'. Pass --base-url.", file=sys.stderr)
            return 1
        ctx = fetch_context_length(
            args.model,
            provider=args.provider,
            base_url=base_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        if ctx is None:
            print(f"{args.model}: <unable to determine context length>")
            return 1
        print(f"{args.model}: {ctx}")
        return 0

    if args.command == "refresh":
        models = _models()
        updated = refresh_auto_context(models)
        if not updated:
            print("No models needed auto-context (all configured or fetch failed).")
            return 0
        for model, ctx in updated.items():
            print(f"{model}: {ctx}")
        if args.write:
            _write_back(updated)
            print("Written back to models.yaml.")
        return 0

    parser.print_help()
    return 0


def _models() -> list[dict[str, Any]]:
    registry = settings.model_registry()
    models = registry.get("models", []) if isinstance(registry, dict) else []
    return [m for m in models if isinstance(m, dict)]


def _entry_for(model: str) -> dict[str, Any]:
    for entry in _models():
        if entry.get("name") == model:
            return entry
    return {}


def _write_back(updated: dict[str, int]) -> None:
    path = settings.config_path("models.yaml")
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    models = data.get("models", [])
    if not isinstance(models, list):
        return
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name in updated:
            entry["context_length"] = updated[name]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    print(f"models.yaml updated ({len(updated)} entries).")


if __name__ == "__main__":
    raise SystemExit(_main())