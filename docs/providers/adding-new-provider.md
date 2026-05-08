# Adding a New Provider

## Overview
AetherMesh supports adding new AI model providers through a consistent adapter pattern.

## Steps

### 1. Create an Adapter
Create `providers/<name>_adapter.py` implementing the provider interface:
```python
from providers.base import BaseProvider

class MyProvider(BaseProvider):
    def chat(self, payload: dict) -> dict:
        ...
    
    def stream(self, payload: dict) -> Iterable[dict]:
        ...
```

### 2. Register Capabilities
Add to `providers/registry.py`:
```python
from providers.registry import ProviderCapabilityEntry, Capability

provider_capability_registry.register(ProviderCapabilityEntry(
    name="my_provider",
    capabilities={Capability.CHAT, Capability.TOOLS},
    health_url="https://api.example.com/health",
    requires_key=True,
    base_url_env="MY_PROVIDER_BASE_URL",
    api_key_env="MY_PROVIDER_API_KEY",
    default_base_url="https://api.example.com/v1",
))
```

### 3. Add Routing Scores
Add capability scores to `runtime/orchestration/routing_engine.py`:
```python
CAPABILITY_PROVIDER_SCORES["chat"]["my_provider"] = 85
```

### 4. Add Cloud Endpoint Config
Add to `CLOUD_PROVIDER_ENDPOINTS` if it's a cloud provider.

### 5. Configure Environment
```
MY_PROVIDER_API_KEY=sk-...
MY_PROVIDER_BASE_URL=https://api.example.com/v1
```
