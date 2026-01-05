# main.py
import base64
import json
import platform
import os
import sys
import importlib.metadata
from typing import Union, List
from pydantic import BaseModel
from fastapi import FastAPI, Response
from sse_starlette.sse import EventSourceResponse
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# ダミーのbitsandbytesモジュールを作成してimportlib.metadata.versionのチェックを回避
class DummyBitsAndBytes:
    __version__ = "0.0.0"  # 適当なバージョンを設定
sys.modules['bitsandbytes'] = DummyBitsAndBytes()

# M5 Macの場合、量子化なしでモデルをロード
is_macos = platform.system() == "Darwin"

if is_macos:
    print("🍎 macOS検出: M5 Mac向け設定を適用します")
    # M5 Macではbitsandbytesが動作しないため、量子化なしでロード
    print("  ⚠️  量子化モデルはbitsandbytesが必要（CUDA専用）のため、量子化なしモデルを使用します")
    model_name = 'openbmb/MiniCPM-Llama3-V-2_5'  # 量子化なしモデル
    # quantization_configをNoneに設定して量子化を無効化
    quantization_config = None
else:
    print("🪟 Windows検出: 量子化モデルを使用します")
    model_name = 'openbmb/MiniCPM-Llama3-V-2_5-int4'  # 量子化モデル
    # BitsAndBytesConfigを明示的に設定して量子化を有効化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

print(f"📦 モデルをロード中: {model_name}")
print("   ⚠️  初回起動時はモデルのダウンロードに時間がかかります")
try:
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,  # 量子化設定を渡す
        torch_dtype=torch.float16 if is_macos else None,  # M5 Macではfloat16を推奨
    )
except Exception as e:
    print(f"❌ モデルのロード中にエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.eval()

class ImageURL(BaseModel):
    url: str = ""

class Content(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None

class Message(BaseModel):
    role: str
    content: list[Content]

class ChatRequest(BaseModel):
    messages: list[Message]
    # OpenAI互換パラメータ（オプション）
    temperature: float = 0.7
    max_tokens: int = 200
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None
    stream: bool = True
    # text-generation-webui互換パラメータ（オプション）
    mode: str = "instruct"
    skip_special_tokens: bool = False
    custom_token_bans: str = ""

class Delta(BaseModel):
    role: str = "assistant"
    content: str = ""

class Choice(BaseModel):
    index: int = 0
    finish_reason: str | None = None
    delta: Delta

class ChatResponse(BaseModel):
    id: str = "chatcmpl-00000"
    object: str = "chat.completions.chunk"
    created: int = 0
    model: str = "MiniCPM-Llama3-V-2_5-int4"
    choices: list[Choice]

app = FastAPI()

def base64_to_image(base64_string):
    # Remove the data URI prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    return image_bytes

def create_image_from_bytes(image_bytes):
    # Create a BytesIO object to handle the image data
    image_stream = BytesIO(image_bytes)

    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    return image

def chat_generator(chatRequest: ChatRequest):
    image = None
    msgs = []

    for message in chatRequest.messages:
        for content in message.content:
            if content.type == "text":
                msgs.append({'role': message.role, 'content': content.text})
            elif content.type == "image_url" and content.image_url and content.image_url.url:
                image_bytes = base64_to_image(content.image_url.url)
                image = create_image_from_bytes(image_bytes).convert('RGB')

    ## if you want to use streaming, please make sure sampling=True and stream=True
    ## the model.chat will return a generator
    # 画像がない場合、モデルのprocessorが空のリストを処理できないため、
    # 適切なサイズのダミー画像を生成する（モデルは448x448を期待）
    if image is None:
        # ダミー画像を生成（モデルが空のリストを処理できないため、448x448の黒画像を使用）
        image = Image.new('RGB', (448, 448), color=(0, 0, 0))
        print("⚠️  画像がないため、ダミー画像を使用します（テキストのみモード）")

    # パラメータをリクエストから取得（デフォルト値を使用）
    temperature = chatRequest.temperature
    max_tokens = chatRequest.max_tokens
    top_p = chatRequest.top_p if chatRequest.top_p < 1.0 else None  # top_p=1.0の場合はNone（無効化）
    stop_strings = chatRequest.stop if chatRequest.stop else []
    stream = chatRequest.stream

    # model.chat()のパラメータを構築
    chat_kwargs = {
        "image": image,
        "msgs": msgs,
        "tokenizer": tokenizer,
        "sampling": True,
        "temperature": temperature,
        "stream": stream,
    }

    # top_pが指定されている場合のみ追加（MiniCPM-Vが対応しているかは不明だが、試してみる）
    if top_p is not None:
        chat_kwargs["top_p"] = top_p

    # max_tokensが指定されている場合、ストリーミング中に制限を適用
    res = model.chat(**chat_kwargs)

    generated_text = ""
    index = 0
    token_count = 0

    for new_text in res:
        # max_tokens制限をチェック
        if max_tokens > 0 and token_count >= max_tokens:
            break

        # stop文字列をチェック
        generated_text += new_text
        should_stop = False
        for stop_str in stop_strings:
            if stop_str in generated_text:
                # stop文字列が見つかった場合、その前までを返す
                stop_index = generated_text.find(stop_str)
                if stop_index >= 0:
                    generated_text = generated_text[:stop_index]
                    should_stop = True
                    break

        if should_stop:
            break

        # トークン数を概算（簡易的に文字数から推定）
        token_count += len(new_text.split())

        print(new_text, flush=True, end='')
        delta = Delta(role="assistant", content=new_text)
        choice = Choice(index=index, finish_reason=None, delta=delta)
        chatResponse = ChatResponse(choices=[choice])
        index += 1
        yield chatResponse.model_dump_json()

    # 最終チャンク（finish_reason="stop"）
    delta = Delta(role="assistant", content="")
    finish_reason = "stop" if (max_tokens > 0 and token_count >= max_tokens) or should_stop else "stop"
    choice = Choice(index=index, finish_reason=finish_reason, delta=delta)
    chatResponse = ChatResponse(choices=[choice])
    yield chatResponse.model_dump_json()


@app.post("/v1/chat/completions")
def chat_completions(chatRequest: ChatRequest):
    # ストリーミングモードの場合
    if chatRequest.stream:
        return EventSourceResponse(chat_generator(chatRequest))
    else:
        # 非ストリーミングモード（同期応答）
        try:
            # 通常のジェネレータを実行して結果を収集
            result_text = ""
            for chunk_json in chat_generator(chatRequest):
                chunk = json.loads(chunk_json)
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        result_text += content

            # 非ストリーミング形式のレスポンスを返す
            from datetime import datetime
            response = {
                "id": "chatcmpl-00000",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": "MiniCPM-Llama3-V-2_5-int4",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # 簡易実装のため0
                    "completion_tokens": 0,  # 簡易実装のため0
                    "total_tokens": 0  # 簡易実装のため0
                }
            }
            return response
        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}, 500
