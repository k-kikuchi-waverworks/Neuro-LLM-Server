#!/bin/bash

# Neuro-LLM-Server 起動スクリプト
# Linux/WSL対応

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# プラットフォーム判定
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "[ERROR] macOSはサポート対象外です（Linux/WSLのみ対応）"
    exit 1
elif [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
    PLATFORM="linux"
else
    echo "[ERROR] 未対応のOSです（Linux/WSLのみ対応）"
    exit 1
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

# プラットフォーム別の環境変数を設定（Linux/WSL）
echo "[OK] Linux/WSL向け設定を適用しました"

echo "[INFO] サーバーを起動中..."
echo "   エンドポイント: http://127.0.0.1:8000"
echo ""

# FastAPIサーバーを起動（python -m uvicornを使用）
python -m uvicorn main:app --host 127.0.0.1 --port 8000
