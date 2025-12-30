# main.py
import base64
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

async def chat_generator(chatRequest: ChatRequest):
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

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        stream=True
    )

    generated_text = ""
    index = 0
    for new_text in res:
        generated_text += new_text
        print(new_text, flush=True, end='')
        delta = Delta(role="assistant", content=new_text)
        choice = Choice(index=index, finish_reason=None, delta=delta)
        chatResponse = ChatResponse(choices=[choice])
        index += 1
        yield chatResponse.model_dump_json()
    delta = Delta(role="assistant", content="")
    choice = Choice(index=index, finish_reason="stop", delta=delta)
    chatResponse = ChatResponse(choices=[choice])
    yield chatResponse.model_dump_json()


@app.post("/v1/chat/completions")
def chat_completions(chatRequest: ChatRequest):
    return EventSourceResponse(chat_generator(chatRequest))
