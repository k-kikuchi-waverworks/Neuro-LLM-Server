#!/bin/bash

# Neuro-LLM-Server セットアップスクリプト
# Mac/Windows対応版

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# プラットフォーム判定
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows"
    echo "[Windows] Neuro-LLM-Server セットアップを開始します..."
    PYTHON_CMD="python"
    VENV_ACTIVATE="venv/Scripts/activate"
else
    PLATFORM="mac"
    echo "[Mac] Neuro-LLM-Server セットアップを開始します..."
    PYTHON_CMD="python3"
    VENV_ACTIVATE="venv/bin/activate"
fi

echo ""

# venvを作成
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    $PYTHON_CMD -m venv venv
    echo "venv created"
fi

# venvをアクティベート
source "$VENV_ACTIVATE"

# pipをアップグレード
echo "[INFO] pipをアップグレード中..."
pip install --upgrade pip

# PyTorchをインストール（プラットフォーム別）
if [ "$PLATFORM" = "mac" ]; then
    echo "[Mac] M5 Mac向けPyTorchをインストール中..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
else
    echo "[Windows] PyTorchをインストール中..."
    pip install torch torchvision torchaudio
fi

# 依存関係をインストール
echo "[INFO] 依存関係をインストール中..."
pip install -r requirements.txt

# fastapi[standard]をインストール（fastapi runコマンド用）
echo "[INFO] fastapi[standard]をインストール中..."
pip install "fastapi[standard]"

echo "[OK] セットアップが完了しました！"
echo ""
echo "起動方法:"
echo "  ./start.sh"
if [ "$PLATFORM" = "windows" ]; then
    echo "  または"
    echo "  source venv/Scripts/activate && uvicorn main:app --host 127.0.0.1 --port 8000"
else
    echo "  または"
    echo "  source venv/bin/activate && fastapi run main.py --port 8000"
fi
