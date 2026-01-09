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
    echo "  [INFO] CPU版PyTorchをインストールします（量子化なしモデルを使用）"
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
else
    echo "[Windows] PyTorchをインストール中..."
    echo "  [INFO] CUDA 12.1対応版PyTorchをインストールします（bitsandbytesで量子化モデルを使用可能）"
    echo "  [TIP] CUDAが利用できない場合は、CPU版がインストールされます"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# 依存関係をインストール
echo "[INFO] 依存関係をインストール中..."
pip install -r requirements.txt

# Windowsでbitsandbytesをインストール（CUDA環境の場合）
if [ "$PLATFORM" = "windows" ]; then
    echo "[Windows] bitsandbytesのインストールを試行中..."
    echo "  [INFO] CUDA環境で量子化モデルを使用するためにbitsandbytesが必要です"
    if pip install bitsandbytes 2>/dev/null; then
        echo "  [OK] bitsandbytesのインストールに成功しました"
    else
        echo "  [WARN] bitsandbytesのインストールに失敗しました（CUDAが利用できない可能性があります）"
        echo "  [TIP] 手動でインストールする場合: pip install bitsandbytes"
    fi
fi

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
