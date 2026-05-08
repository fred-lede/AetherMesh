from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


class AetherMeshClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8002", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def chat(self, model: str, message: str, stream: bool = False, max_tokens: int = 256) -> dict[str, Any] | None:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": message}],
            "stream": stream,
        }
        resp = self.session.post(f"{self.base_url}/v1/messages", json=payload, stream=stream, timeout=60)
        resp.raise_for_status()
        if stream:
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data = decoded[6:]
                        if data == "[DONE]":
                            break
                        print(json.dumps(json.loads(data), indent=2))
            return None
        return resp.json()

    def list_models(self) -> list[dict[str, Any]]:
        resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    def health(self) -> dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="AetherMesh CLI")
    parser.add_argument("--url", default="http://127.0.0.1:8002", help="Router base URL")
    parser.add_argument("--api-key", default="", help="API key")
    sub = parser.add_subparsers(dest="command", required=True)

    chat_parser = sub.add_parser("chat")
    chat_parser.add_argument("model", help="Model name")
    chat_parser.add_argument("message", help="Message to send")
    chat_parser.add_argument("--max-tokens", type=int, default=256)
    chat_parser.add_argument("--stream", action="store_true")

    sub.add_parser("models")
    sub.add_parser("health")

    args = parser.parse_args()
    client = AetherMeshClient(base_url=args.url, api_key=args.api_key)

    if args.command == "chat":
        result = client.chat(args.model, args.message, stream=args.stream, max_tokens=args.max_tokens)
        if result and not args.stream:
            content = result.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block.get("text", ""))
    elif args.command == "models":
        for m in client.list_models():
            print(f"{m['name']:40s} {m.get('provider', '')}")
    elif args.command == "health":
        print(json.dumps(client.health(), indent=2))


if __name__ == "__main__":
    main()
