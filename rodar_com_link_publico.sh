#!/usr/bin/env bash
# Plano C: roda o app no seu PC e cria um link público temporário (Cloudflare, sem conta).
# Uso: ./rodar_com_link_publico.sh   (Ctrl+C encerra tudo)
set -euo pipefail
cd "$(dirname "$0")"
PY="../prototipo/.venv/bin/streamlit"
"$PY" run app.py --server.port 8501 --server.headless true &
APP=$!
trap 'kill $APP 2>/dev/null' EXIT
sleep 3
echo ">>> O link público aparece abaixo (linha com trycloudflare.com). Envie-o ao professor."
cloudflared tunnel --url http://localhost:8501
