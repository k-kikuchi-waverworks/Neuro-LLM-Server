#!/bin/bash

# Neuro-LLM-Server 起動スクリプト
# M5 Mac対応版

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Neuro-LLM-Server を起動します..."
echo ""

# venvをアクティベート
if [ ! -d "venv" ]; then
    echo "❌ venvが見つかりません。まず ./setup.sh を実行してください"
    exit 1
fi

source venv/bin/activate

# M5 Mac向けの環境変数を設定
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export USE_MPS=1

echo "🍎 M5 Mac向けGPU設定を適用しました"
echo "📡 サーバーを起動中..."
echo "   エンドポイント: http://127.0.0.1:8000"
echo ""

# FastAPIサーバーを起動（uvicornを直接使用）
uvicorn main:app --host 127.0.0.1 --port 8000
