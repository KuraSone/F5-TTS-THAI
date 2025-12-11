import os
from typing import Annotated

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import asyncio
import itertools
import httpx
import io
import traceback
import soundfile as sf
from contextlib import asynccontextmanager

import soundfile as sf  # type: ignore
import uvicorn
from pydantic import BaseModel, Field
from fastapi import APIRouter, Response, Request, Depends, FastAPI, Header

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything
from f5_tts.cleantext.th_normalize import normalize_text
from f5_tts.infer.infer_gradio import *

from src.f5_tts.f5_tts_webui import infer_tts


audio_cache: dict[str, bytes] = {}


class Data(BaseModel):
    text: str
    audio_paths: list[str]
    language: str | None
    ref_audio: list[bytes] = []


tts_th: None




@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/tts_url")
async def tts(data: Data, request: Request, x_api_key: Annotated[str | None, Header()] = None) -> Response:
    try:
        if x_api_key != 'domeker-ai-ZG9tZWtlci1haQ==':
            return Response(content="Invalid API key", status_code=401)
        audio_url = data.audio_paths[0]
        if audio_url in audio_cache:
            audio_bytes = audio_cache[audio_url]
            data.ref_audio.append(audio_bytes)
        else:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(audio_url)
                audio_bytes = response.content
                audio_cache[audio_url] = audio_bytes
                data.ref_audio.append(audio_bytes)

        (sr, wav), _, _, _ = infer_tts(data.ref_audio[0], '', data.text)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='wav')
            wav_bytes: bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        traceback.print_exc()
        return Response(content=str(e), status_code=500)

    finally:
        pass


if __name__ == "__main__":
    uvicorn.run(app='app:app', host='0.0.0.0', port=3392)
