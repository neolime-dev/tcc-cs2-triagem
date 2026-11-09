#!/usr/bin/env bash
# Plano C: roda o Terminal de triagem no seu PC e cria um link público temporário (Cloudflare, sem conta).
# Uso: ./rodar_com_link_publico.sh   (Ctrl+C encerra tudo)
set -euo pipefail
cd "$(dirname "$0")"
../prototipo/.venv/bin/python server.py --port 8765 &
APP=$!
trap 'kill $APP 2>/dev/null' EXIT
echo ">>> Aguardando o servidor ficar pronto..."
until curl -s http://127.0.0.1:8765/api/health >/dev/null; do sleep 1; done
echo ">>> O link público aparece abaixo (linha com trycloudflare.com). Envie-o ao professor."
cloudflared tunnel --url http://localhost:8765
