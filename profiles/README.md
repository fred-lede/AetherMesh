# Deployment Profiles

Use these profile templates to avoid mixing control-plane and worker-node settings.

- `control-plane/.env.example`: recommended base env for the central host
- `worker-node/.env.example`: recommended base env for remote GPU nodes
- `worker-node/ollama-gpu0.env.example` and `worker-node/ollama-gpu1.env.example`: env files for dual Ollama workers

Copy only the profile that matches the machine role.
