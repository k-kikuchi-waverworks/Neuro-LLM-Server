#!/bin/bash

# Neuro-LLM-Server セットアップスクリプト
# M5 Mac対応版

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Neuro-LLM-Server セットアップを開始します..."
echo ""

# venvを作成
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
    echo "venv created"
fi

# venvをアクティベート
source venv/bin/activate

# pipをアップグレード
echo "📦 pipをアップグレード中..."
pip install --upgrade pip

# M5 Mac向けPyTorchをインストール
echo "🍎 M5 Mac向けPyTorchをインストール中..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# 依存関係をインストール
echo "📦 依存関係をインストール中..."
pip install -r requirements.txt

# fastapi[standard]をインストール（fastapi runコマンド用）
echo "📦 fastapi[standard]をインストール中..."
pip install "fastapi[standard]"

echo "✅ セットアップが完了しました！"
echo ""
echo "起動方法:"
echo "  ./start.sh"
echo "  または"
echo "  source venv/bin/activate && fastapi run main.py --port 8000"


