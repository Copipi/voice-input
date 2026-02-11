#!/bin/bash
set -e

echo "=== voice-input server ==="

# Start Ollama in background
echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready."
        break
    fi
    sleep 1
done

# Pull default models if not present
LLM_MODEL="${LLM_MODEL:-gpt-oss:20b}"
VISION_MODEL="${VISION_MODEL:-qwen3-vl:8b-instruct}"

echo "Checking models..."
ollama pull "$LLM_MODEL" 2>/dev/null || echo "Warning: could not pull $LLM_MODEL"
ollama pull "$VISION_MODEL" 2>/dev/null || echo "Warning: could not pull $VISION_MODEL"

echo "Starting WebSocket server..."
exec python3 ws_server.py "$@"
