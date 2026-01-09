#!/bin/bash

# Neuro-LLM-Server 起動スクリプト
# Mac/Windows対応版

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# プラットフォーム判定
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows"
    VENV_ACTIVATE="venv/Scripts/activate"
else
    PLATFORM="mac"
    VENV_ACTIVATE="venv/bin/activate"
fi

echo "[INFO] Neuro-LLM-Server を起動します..."
echo ""

# venvをアクティベート
if [ ! -d "venv" ]; then
    echo "[ERROR] venvが見つかりません。まず ./setup.sh を実行してください"
    exit 1
fi

source "$VENV_ACTIVATE"

# M5 Mac向けの環境変数を設定（Macの場合のみ）
if [ "$PLATFORM" = "mac" ]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export USE_MPS=1
    echo "[OK] M5 Mac向けGPU設定を適用しました"
else
    echo "[OK] Windows向け設定を適用しました"
fi

echo "[INFO] サーバーを起動中..."
echo "   エンドポイント: http://127.0.0.1:8000"
echo ""

# FastAPIサーバーを起動（uvicornを直接使用）
uvicorn main:app --host 127.0.0.1 --port 8000
