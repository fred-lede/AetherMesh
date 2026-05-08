from __future__ import annotations

from runtime.orchestration.capabilities import required_anthropic_capabilities, required_openai_capabilities
from runtime.orchestration.openai_handler import RouterService
from runtime.orchestration.routing_engine import ModelRoutingEngine


def test_anthropic_required_capabilities_include_audio_vision_tools_thinking() -> None:
    required = required_anthropic_capabilities(
        {
            "model": "glm-4.7-flash:q4_K_M",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{"name": "Search", "input_schema": {"type": "object"}}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image", "source": {"type": "base64", "data": "img"}},
                        {"type": "audio", "source": {"type": "base64", "data": "aud"}},
                    ],
                }
            ],
        }
    )

    assert required == {"chat", "thinking", "tools", "vision", "audio"}


def test_openai_required_capabilities_include_nested_tool_result_image() -> None:
    required = required_openai_capabilities(
        {
            "model": "x",
            "messages": [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,img"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    assert required == {"chat", "vision"}


def test_routing_skips_exact_local_model_when_required_capability_missing() -> None:
    engine = ModelRoutingEngine()
    registry = [
        {
            "name": "text-only",
            "provider": "ollama",
            "worker_bindings": [{"node_id": "node-01", "port": 11434}],
            "capabilities": ["chat", "tools"],
        },
        {
            "name": "vision-model",
            "provider": "ollama",
            "worker_bindings": [{"node_id": "node-01", "port": 11434}],
            "capabilities": ["chat", "tools", "vision"],
        },
    ]

    decision = engine.route(
        model="text-only",
        required_capabilities=["chat", "vision"],
        registry_models=registry,
    )

    assert decision.provider == "ollama"
    assert decision.model == "vision-model"
    assert any(
        "capability_fallback text-only -> vision-model" in rule
        for rule in decision.rules_applied
    )


def test_openai_resolver_uses_capability_fallback_before_dispatch() -> None:
    service = RouterService()
    service.registry = {
        "models": [
            {
                "name": "text-only",
                "provider": "ollama",
                "worker_bindings": [{"node_id": "node-01", "port": 11434}],
                "capabilities": ["chat"],
            },
            {
                "name": "vision-model",
                "provider": "ollama",
                "worker_bindings": [{"node_id": "node-01", "port": 11434}],
                "capabilities": ["chat", "vision"],
            },
        ]
    }
    payload = {
        "model": "text-only",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img"},
                    }
                ],
            }
        ],
    }

    provider, worker = service._resolve_provider_and_worker(payload, allow_queue=False)

    assert provider == "ollama"
    assert payload["model"] == "vision-model"
    assert worker == {"base_url": "http://127.0.0.1:11434"}
