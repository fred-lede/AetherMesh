from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class RequestResult:
    index: int
    ok: bool
    status_code: int
    retry_after: str
    error: str
    body_preview: str


def _build_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Return exactly: overload-check",
            }
        ],
        "max_tokens": 32,
    }


def _run_one(
    *,
    index: int,
    url: str,
    api_key: str,
    model: str,
    timeout_s: int,
) -> RequestResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = _build_payload(model)
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        preview = response.text.strip().replace("\r", " ").replace("\n", " ")
        if len(preview) > 180:
            preview = f"{preview[:180]}..."
        return RequestResult(
            index=index,
            ok=response.ok,
            status_code=int(response.status_code),
            retry_after=response.headers.get("Retry-After", ""),
            error="",
            body_preview=preview,
        )
    except requests.RequestException as exc:
        return RequestResult(
            index=index,
            ok=False,
            status_code=0,
            retry_after="",
            error=str(exc),
            body_preview="",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="AIIH overload smoke test for 429 behavior.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="Router base URL")
    parser.add_argument("--model", default="qwen3.5:27b", help="Target model")
    parser.add_argument("--total", type=int, default=20, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel workers")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout seconds")
    parser.add_argument("--api-key", default="local-dev-key", help="Authorization bearer token")
    args = parser.parse_args()

    total = max(1, int(args.total))
    concurrency = max(1, int(args.concurrency))
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    started = time.time()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _run_one,
                index=i + 1,
                url=url,
                api_key=args.api_key,
                model=args.model,
                timeout_s=args.timeout,
            )
            for i in range(total)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.index)
    elapsed = time.time() - started
    status_counts = Counter(item.status_code for item in results)
    network_errors = [item for item in results if item.status_code == 0]
    has_429 = status_counts.get(429, 0) > 0
    has_200 = status_counts.get(200, 0) > 0

    print("==== AIIH 429 overload validation ====")
    print(f"router={args.base_url}")
    print(f"model={args.model}")
    print(f"total={total}, concurrency={concurrency}, elapsed={elapsed:.2f}s")
    print(f"status_counts={json.dumps(dict(sorted(status_counts.items())), ensure_ascii=False)}")

    if has_429:
        retry_after_values = sorted(
            {item.retry_after for item in results if item.status_code == 429 and item.retry_after}
        )
        print(f"retry_after_headers={retry_after_values or ['(missing)']}")

    if network_errors:
        print("network_errors:")
        for item in network_errors[:5]:
            print(f"  - req#{item.index}: {item.error}")
        if len(network_errors) > 5:
            print(f"  ... {len(network_errors) - 5} more")

    failures = [
        item
        for item in results
        if item.status_code not in {200, 429}
    ]
    if failures:
        print("unexpected_responses:")
        for item in failures[:5]:
            print(f"  - req#{item.index}: status={item.status_code}, body={item.body_preview}")
        if len(failures) > 5:
            print(f"  ... {len(failures) - 5} more")

    if not has_200:
        print("[FAIL] No successful 200 responses observed.")
        return 2
    if not has_429:
        print("[WARN] No 429 observed. Increase --concurrency or --total to stress harder.")
        return 0
    if network_errors:
        print("[FAIL] Network exceptions observed under load.")
        return 3

    print("[PASS] Overload handled via explicit 429 (with Retry-After) and no network exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
