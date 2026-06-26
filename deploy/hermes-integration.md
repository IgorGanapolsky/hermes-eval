# Pointing Hermes (or any OpenAI-compatible agent) at the gateway

The gateway is a drop-in OpenAI-compatible endpoint, so any client points at it with one provider block.

## 1. Run the proxy persistently (launchd, macOS)

```bash
cp deploy/com.igor.hermes-litellm.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.igor.hermes-litellm.plist   # RunAtLoad + KeepAlive
curl -s http://127.0.0.1:4010/health/liveliness                          # -> "I'm alive!"
```

## 2. Add the provider to Hermes (`~/.hermes/config.yaml`) — additive, test before flipping

```yaml
providers:
  litellm-gateway:
    name: LiteLLM Gateway (local fleet via proxy :4010)
    api: http://127.0.0.1:4010/v1
    base_url: http://127.0.0.1:4010/v1
    transport: chat_completions
    default_model: hermes-local-fast
    model: hermes-local-fast
    api_key: sk-hermes-local-dev
    discover_models: true
```

Test without changing the default:

```bash
hermes -z "ping" --provider custom:litellm-gateway -m hermes-local-fast --yolo
```

## 3. Flip the default (only after the test passes)

```yaml
model:
  default: hermes-local-fast
  provider: custom:litellm-gateway
```

**Keep a direct-Ollama fallback** so a dead proxy never kills the agent:

```yaml
fallback_providers:
  - {provider: custom:ollama-local-64k, model: qwen2.5:3b-64k, base_url: 'http://127.0.0.1:11434/v1', api_mode: chat_completions, context_length: 65536}
```

Now every Hermes call flows through the gateway and is appended to
`~/.hermes/litellm-logs/traffic.jsonl` — your golden-set / drift feed. Curate those traces into
`eval/golden.jsonl` and the loop is closed.

To revert: `cp ~/.hermes/config.yaml.bak.* ~/.hermes/config.yaml` and `launchctl unload` the plist.
