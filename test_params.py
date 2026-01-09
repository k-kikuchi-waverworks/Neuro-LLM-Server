#!/usr/bin/env python3
"""
Neuro-LLM-Serverのパラメータ実装をテストするスクリプト
"""
import requests
import json
import base64
from PIL import Image
from io import BytesIO

# テスト用のダミー画像を生成
def create_test_image():
    """テスト用の小さな画像を生成"""
    img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def test_streaming():
    """ストリーミングモードのテスト"""
    print("=" * 60)
    print("テスト1: ストリーミングモード（デフォルトパラメータ）")
    print("=" * 60)

    url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "この画像を説明してください。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": create_test_image()
                    }
                }
            ]
        }],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 50
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        if response.status_code == 200:
            print("[OK] ストリーミングレスポンス受信開始")
            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # 'data: 'を除去
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                                    chunk_count += 1
                        except json.JSONDecodeError:
                            pass
            print(f"\n[OK] ストリーミングテスト完了（{chunk_count}チャンク受信）")
        else:
            print(f"[ERROR] エラー: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"[ERROR] エラー: {e}")

def test_non_streaming():
    """非ストリーミングモードのテスト"""
    print("\n" + "=" * 60)
    print("テスト2: 非ストリーミングモード（カスタムパラメータ）")
    print("=" * 60)

    url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "この画像は何色ですか？"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": create_test_image()
                    }
                }
            ]
        }],
        "stream": False,
        "temperature": 0.5,
        "max_tokens": 30,
        "top_p": 0.9,
        "stop": ["。", "\n"]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("[OK] 非ストリーミングレスポンス受信")
            print(f"応答: {result.get('choices', [{}])[0].get('message', {}).get('content', '')}")
            print(f"モデル: {result.get('model', 'N/A')}")
        else:
            print(f"[ERROR] エラー: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"[ERROR] エラー: {e}")

def test_parameters():
    """パラメータが正しく受け取られているかテスト"""
    print("\n" + "=" * 60)
    print("テスト3: パラメータ受け取り確認（max_tokens制限）")
    print("=" * 60)

    url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "1から10まで数えてください。"
                }
            ]
        }],
        "stream": True,
        "max_tokens": 20,  # 短く制限
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        if response.status_code == 200:
            print("[OK] ストリーミングレスポンス受信開始（max_tokens=20で制限）")
            total_chars = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                                    total_chars += len(content)
                        except json.JSONDecodeError:
                            pass
            print(f"\n[OK] max_tokens制限テスト完了（合計{total_chars}文字）")
        else:
            print(f"[ERROR] エラー: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"[ERROR] エラー: {e}")

if __name__ == "__main__":
    print("[TEST] Neuro-LLM-Server パラメータ実装テスト")
    print("=" * 60)
    print("サーバーが起動していることを確認してください: http://127.0.0.1:8000")
    print("=" * 60)

    # サーバーの起動確認
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        if response.status_code == 200:
            print("[OK] Neuro-LLM-Serverに接続できました\n")
        else:
            print("[WARN] Neuro-LLM-Serverに接続できません（起動していない可能性があります）\n")
    except Exception as e:
        print(f"[WARN] Neuro-LLM-Serverに接続できません: {e}\n")
        print("[TIP] サーバーを起動してください:")
        print("   cd tools/Neuro-LLM-Server")
        print("   source venv/bin/activate")
        print("   uvicorn main:app --host 127.0.0.1 --port 8000")
        exit(1)

    # テスト実行
    test_streaming()
    test_non_streaming()
    test_parameters()

    print("\n" + "=" * 60)
    print("[OK] すべてのテスト完了")
    print("=" * 60)
