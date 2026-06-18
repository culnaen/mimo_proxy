# mimo_proxy

OpenAI-compatible proxy server for [MiMo Code](https://github.com/XiaomiMiMo/MiMo-Code) CLI.

Exposes standard OpenAI API endpoints and forwards requests to the local `mimo` agent.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (OpenAI-compatible) |
| `POST` | `/v1/messages` | Messages (Anthropic-compatible) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/health` | Health check |

## Quick Start

```bash
# Install MiMo Code (if not installed)
# See: https://github.com/XiaomiMiMo/MiMo-Code

# Build and run
go build -o mimo_proxy .
export no_proxy=localhost,127.0.0.1
./mimo_proxy
# Listening on :8080
```

## Usage

### curl

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mimo-code",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="mimo-code",
    messages=[{"role": "user", "content": "hello"}]
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="mimo-code",
    messages=[{"role": "user", "content": "hello"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Requirements

- Go 1.21+
- `mimo` CLI in PATH ([MiMo Code](https://github.com/XiaomiMiMo/MiMo-Code))

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MIMO_LISTEN_ADDR` | `0.0.0.0:8080` | Listen address |
| `MIMO_BINARY` | `mimo` | Path to mimo binary |
| `MIMO_MODEL` | `mimo-code` | Model name reported to clients |
| `MIMO_MODELS_CONFIG` | `~/.mimocode/models.json` | Custom models config path |

## Notes

- If behind an HTTP proxy, set `no_proxy=localhost,127.0.0.1` before running
- Timeout per request: 300s
- Supports both string and array `content` formats (multimodal)
