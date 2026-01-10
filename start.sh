#!/bin/bash

# Neuro-LLM-Server 起動スクリプト
# Mac/Linux対応版

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

# venvをアクティベート
if [ ! -d "venv" ]; then
    echo "[ERROR] venvが見つかりません。まず ./setup.sh を実行してください"
    exit 1
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "[ERROR] venv/bin/activateが見つかりません"
    echo "  [TIP] venvを再作成する場合: ./setup.sh"
    exit 1
fi

source "$VENV_ACTIVATE"

# 重要な依存関係をチェック
echo "[INFO] 依存関係を確認中..."
MISSING_DEPS=0

if ! python -c "import sse_starlette" 2>/dev/null; then
    echo "  [ERROR] sse-starletteがインストールされていません"
    echo "  [TIP] 以下のコマンドでインストールしてください:"
    echo "    source venv/bin/activate"
    echo "    pip install sse-starlette==2.1.0"
    MISSING_DEPS=1
fi

if ! python -c "import fastapi" 2>/dev/null; then
    echo "  [ERROR] fastapiがインストールされていません"
    echo "  [TIP] 以下のコマンドでインストールしてください:"
    echo "    source venv/bin/activate"
    echo "    pip install fastapi==0.111.0"
    MISSING_DEPS=1
fi

if ! python -c "import uvicorn" 2>/dev/null; then
    echo "  [ERROR] uvicornがインストールされていません"
    echo "  [TIP] 以下のコマンドでインストールしてください:"
    echo "    source venv/bin/activate"
    echo "    pip install 'fastapi[standard]'"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo "[ERROR] 必要な依存関係が不足しています"
    echo "  [TIP] セットアップを再実行する場合: ./setup.sh"
    exit 1
fi

echo "  [OK] すべての依存関係がインストールされています"

# M5 Mac向けの環境変数を設定（Macの場合のみ）
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

# FastAPIサーバーを起動（uvicornを直接使用）
uvicorn main:app --host 127.0.0.1 --port 8000
