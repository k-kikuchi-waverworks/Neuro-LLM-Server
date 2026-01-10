#!/bin/bash

# Neuro-LLM-Server セットアップスクリプト
# Mac/Linux対応

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# プラットフォーム判定
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="mac"
    echo "[Mac] Neuro-LLM-Server セットアップを開始します..."
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
    PLATFORM="linux"
    echo "[Linux/WSL] Neuro-LLM-Server セットアップを開始します..."
    PYTHON_CMD="python3"
else
    PLATFORM="mac"
    echo "[Mac] Neuro-LLM-Server セットアップを開始します（デフォルト）..."
    PYTHON_CMD="python3"
fi

VENV_ACTIVATE="venv/bin/activate"

echo ""

# venvを作成
if [ ! -d "venv" ]; then
    echo "[INFO] venvを作成中..."
    "$PYTHON_CMD" -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] venvの作成に失敗しました"
        exit 1
    fi
    echo "[OK] venvを作成しました"
else
    echo "[INFO] venvは既に存在します"
fi

# venvをアクティベート
echo "[INFO] venvをアクティベート中..."
source "$VENV_ACTIVATE"
if [ $? -ne 0 ]; then
    echo "[ERROR] venvのアクティベートに失敗しました"
    exit 1
fi
echo "[OK] venvをアクティベートしました"

# pipをアップグレード
echo "[INFO] pipをアップグレード中..."
pip install --upgrade pip

# PyTorchをインストール（プラットフォーム別）
if [ "$PLATFORM" = "mac" ]; then
    echo "[Mac] PyTorchをインストール中..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
else
    echo "[Linux/WSL] PyTorchをインストール中..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# 依存関係をインストール
echo "[INFO] 依存関係をインストール中..."
pip install -r requirements.txt

# Linux/WSLでbitsandbytesをインストール（CUDA環境の場合）
if [ "$PLATFORM" = "linux" ]; then
    echo "[Linux/WSL] bitsandbytesのインストールを試行中..."
    pip install bitsandbytes 2>/dev/null || echo "  [WARN] bitsandbytesのインストールに失敗しました（CUDAが利用できない可能性があります）"
fi

# fastapi[standard]をインストール（uvicornを含む）
echo "[INFO] fastapi[standard]をインストール中..."
pip install "fastapi[standard]"

# uvicornがインストールされているか確認
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "[WARN] uvicornがインストールされていません。明示的にインストールします..."
    pip install uvicorn[standard]
fi

echo "[OK] セットアップが完了しました！"
echo ""
echo "起動方法:"
echo "  ./start.sh"
echo "  または"
echo "  source venv/bin/activate && uvicorn main:app --host 127.0.0.1 --port 8000"
