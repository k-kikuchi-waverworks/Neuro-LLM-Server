#!/bin/bash

# Neuro-LLM-Server 起動スクリプト
# Mac/Linux対応

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# プラットフォーム判定
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="mac"
elif [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
    PLATFORM="linux"
else
    PLATFORM="mac"
fi

VENV_ACTIVATE="venv/bin/activate"

echo "[INFO] Neuro-LLM-Server を起動します..."
echo ""

# venvをチェック
if [ ! -d "venv" ] || [ ! -f "$VENV_ACTIVATE" ]; then
    echo "[ERROR] venvが見つかりません。まず ./setup.sh を実行してください"
    exit 1
fi

# venvをアクティベート
source "$VENV_ACTIVATE"

# プラットフォーム別の環境変数を設定
if [ "$PLATFORM" = "mac" ]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export USE_MPS=1
    echo "[OK] M5 Mac向けGPU設定を適用しました"
else
    echo "[OK] Linux/WSL向け設定を適用しました"
fi

echo "[INFO] サーバーを起動中..."
echo "   エンドポイント: http://127.0.0.1:8000"
echo ""

# FastAPIサーバーを起動（python -m uvicornを使用）
python -m uvicorn main:app --host 127.0.0.1 --port 8000
