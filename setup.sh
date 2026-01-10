#!/bin/bash

# Neuro-LLM-Server セットアップスクリプト
# Mac/Linux対応版

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

    # WSL環境ではWindowsのpyenv-winパスを除外してLinux用のPythonを使用
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/mnt/c/.*pyenv' | grep -v '/mnt/c/.*\.pyenv' | tr '\n' ':' | sed 's/:$//')

    # システムのpython3を明示的に使用
    if [ -f "/usr/bin/python3" ]; then
        PYTHON_CMD="/usr/bin/python3"
    elif command -v python3 >/dev/null 2>&1 && ! python3 --version 2>&1 | grep -q "cannot execute"; then
        PYTHON_CMD=$(which python3)
    else
        PYTHON_CMD="python3"
    fi
    echo "  [INFO] Python3コマンド: $PYTHON_CMD"

    # pythonコマンドのエイリアスも設定（python3へのシンボリックリンクがない場合に備えて）
    if [ ! -f "/usr/bin/python" ] && [ "$PYTHON_CMD" = "/usr/bin/python3" ]; then
        # pythonコマンドが存在しない場合は、python3を使用
        alias python="$PYTHON_CMD" 2>/dev/null || true
    fi
else
    PLATFORM="mac"
    echo "[Mac] Neuro-LLM-Server セットアップを開始します（デフォルト）..."
    PYTHON_CMD="python3"
fi

VENV_ACTIVATE="venv/bin/activate"

echo ""

# venvモジュールが利用可能か確認
if ! "$PYTHON_CMD" -m venv --help >/dev/null 2>&1; then
    echo "[ERROR] venvモジュールが利用できません"
    echo ""
    if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
        PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        echo "[INFO] 以下のコマンドでvenvモジュールをインストールしてください:"
        echo "  sudo apt update"
        echo "  sudo apt install python${PYTHON_VERSION}-venv"
        echo ""
        echo "または、python3-venvパッケージをインストール:"
        echo "  sudo apt install python3-venv"
    else
        echo "[INFO] Pythonのvenvモジュールがインストールされていない可能性があります"
    fi
    exit 1
fi

# venvを作成または再作成
# Windows環境で作成されたvenvの可能性があるため、WSL環境では再作成を推奨
if [ ! -d "venv" ]; then
    echo "[INFO] venvを作成中..."
    "$PYTHON_CMD" -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] venvの作成に失敗しました"
        exit 1
    fi
    echo "[OK] venvを作成しました"
elif [ ! -f "$VENV_ACTIVATE" ]; then
    echo "[WARN] venvは存在しますが、activateファイルが見つかりません"
    echo "[INFO] venvを再作成します..."
    rm -rf venv
    "$PYTHON_CMD" -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] venvの再作成に失敗しました"
        exit 1
    fi
    echo "[OK] venvを再作成しました"
elif [ "$PLATFORM" = "linux" ] && [ -d "venv/Lib" ]; then
    # Windows環境で作成されたvenv（Libディレクトリが存在）の場合は再作成
    echo "[WARN] Windows環境で作成されたvenvが検出されました"
    echo "[INFO] WSL環境用にvenvを再作成します..."
    rm -rf venv
    "$PYTHON_CMD" -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] venvの再作成に失敗しました"
        exit 1
    fi
    echo "[OK] venvを再作成しました"
else
    echo "[INFO] venvは既に存在します"
fi

# venvをアクティベート
echo "[INFO] venvをアクティベート中..."
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
    if [ $? -ne 0 ]; then
        echo "[ERROR] venvのアクティベートに失敗しました"
        exit 1
    fi
    echo "[OK] venvをアクティベートしました"
else
    echo "[ERROR] venv/bin/activateが見つかりません"
    exit 1
fi

# WSL/Linux環境でシステム依存関係をチェック（sentencepieceなど）
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
    echo "[INFO] システム依存関係をチェック中..."

    # cmakeとC++コンパイラをチェック（sentencepieceのビルドに必要）
    if ! command -v cmake >/dev/null 2>&1; then
        echo "[WARN] cmakeがインストールされていません"
        echo "[INFO] sentencepieceのビルドに必要です"
        echo ""
        echo "以下のコマンドでインストールしてください:"
        echo "  sudo apt update"
        echo "  sudo apt install -y cmake build-essential"
        echo ""
        echo "[INFO] セットアップを続行しますが、sentencepieceのインストールが失敗する可能性があります"
    else
        echo "  [OK] cmakeがインストールされています"
    fi

    # build-essential（C++コンパイラ）をチェック
    if ! command -v g++ >/dev/null 2>&1; then
        echo "[WARN] g++（C++コンパイラ）がインストールされていません"
        echo "[INFO] sentencepieceのビルドに必要です"
        echo ""
        echo "以下のコマンドでインストールしてください:"
        echo "  sudo apt install -y build-essential"
        echo ""
    else
        echo "  [OK] g++（C++コンパイラ）がインストールされています"
    fi
fi

# pipをアップグレード
echo "[INFO] pipをアップグレード中..."
pip install --upgrade pip

# PyTorchをインストール（プラットフォーム別）
if [ "$PLATFORM" = "mac" ]; then
    echo "[Mac] M5 Mac向けPyTorchをインストール中..."
    echo "  [INFO] CPU版PyTorchをインストールします（量子化なしモデルを使用）"
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
else
    echo "[Linux/WSL] PyTorchをインストール中..."
    echo "  [INFO] CUDA 12.1対応版PyTorchをインストールします（bitsandbytesで量子化モデルを使用可能）"
    echo "  [TIP] CUDAが利用できない場合は、CPU版がインストールされます"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# 依存関係をインストール
echo "[INFO] 依存関係をインストール中..."
INSTALL_OUTPUT=$(pip install -r requirements.txt 2>&1)
INSTALL_STATUS=$?

if [ $INSTALL_STATUS -eq 0 ]; then
    echo "  [OK] 依存関係のインストールが完了しました"
else
    echo "  [WARN] 依存関係のインストールに一部失敗しました（exit code: $INSTALL_STATUS）"

    # sentencepieceのビルドエラーをチェック
    if echo "$INSTALL_OUTPUT" | grep -q "sentencepiece\|cmake: not found"; then
        echo ""
        echo "[INFO] sentencepieceのビルドエラーが検出されました"
        if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
            echo "[INFO] 以下のシステム依存関係が必要です:"
            echo "  sudo apt update"
            echo "  sudo apt install -y cmake build-essential"
            echo ""
            echo "インストール後、以下のコマンドでsentencepieceを再インストール:"
            echo "  source venv/bin/activate"
            echo "  pip install sentencepiece==0.1.99"
        fi
        echo ""
        echo "[INFO] 他の依存関係はインストールされている可能性があります"
        echo "[INFO] セットアップを続行します..."
    else
        echo "  [ERROR] 予期しないエラーが発生しました"
        echo "  [TIP] 手動でインストールする場合:"
        echo "    source venv/bin/activate"
        echo "    pip install -r requirements.txt"
    fi
fi

# 重要な依存関係を個別に確認
echo "[INFO] 重要な依存関係を確認中..."
MISSING_DEPS=0

if ! python -c "import sse_starlette" 2>/dev/null; then
    echo "  [WARN] sse-starletteがインストールされていません"
    echo "  [INFO] 手動でインストールします..."
    if pip install sse-starlette==2.1.0; then
        echo "  [OK] sse-starletteのインストールに成功しました"
    else
        echo "  [ERROR] sse-starletteのインストールに失敗しました"
        MISSING_DEPS=1
    fi
else
    echo "  [OK] sse-starletteがインストールされています"
fi

if ! python -c "import fastapi" 2>/dev/null; then
    echo "  [WARN] fastapiがインストールされていません"
    MISSING_DEPS=1
else
    echo "  [OK] fastapiがインストールされています"
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo "[ERROR] 必要な依存関係が不足しています"
    echo "  [TIP] 手動でインストールする場合:"
    echo "    source venv/bin/activate"
    echo "    pip install sse-starlette==2.1.0 fastapi==0.111.0"
    exit 1
fi

# Linux/WSLでbitsandbytesをインストール（CUDA環境の場合）
if [ "$PLATFORM" = "linux" ]; then
    echo "[Linux/WSL] bitsandbytesのインストールを試行中..."
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
if pip install "fastapi[standard]"; then
    echo "  [OK] fastapi[standard]のインストールが完了しました"
else
    echo "  [WARN] fastapi[standard]のインストールに失敗しました（uvicornは既にインストールされている可能性があります）"
fi

echo "[OK] セットアップが完了しました！"
echo ""
echo "起動方法:"
echo "  ./start.sh"
echo "  または"
echo "  source venv/bin/activate && uvicorn main:app --host 127.0.0.1 --port 8000"
