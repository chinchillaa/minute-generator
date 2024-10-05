import openai
from typing import List, Optional
from io import BytesIO
import os
from streamlit.runtime.uploaded_file_manager import (
    UploadedFile,
)  # StreamlitのUploadedFileクラス
import streamlit as st
import tempfile


# def transcribe_audio(audio_file: Optional[UploadedFile], open_ai_api_key: str) -> str:
#     """
#     Transcribe an audio file using the Whisper model in OpenAI's API.
#     """
#     openai.api_key = open_ai_api_key
#     transcript = openai.Audio.transcribe("whisper-1", audio_file)

#     return transcript["text"]


def transcribe_audio(
    audio_file: Optional[UploadedFile], open_ai_api_key: str, chunk_size: int
) -> str:
    """UploadedFile型のオーディオファイルをWhisperで文字起こし"""
    openai.api_key = open_ai_api_key

    # アップロードされたファイルが25MBを超える場合に分割
    chunks = split_file(audio_file, chunk_size)

    # 元のファイルの拡張子を取得
    _, ext = os.path.splitext(audio_file.name)

    # 各チャンクを順番に処理して、すべての結果をつなげる
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        st.write(f"Transcribing chunk {i+1}/{len(chunks)}")
        # 一時ファイルを作成し、OpenAI APIに渡す
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_file.write(
                chunk.getbuffer()
            )  # `BytesIO`オブジェクトから一時ファイルに書き込み
            temp_file.flush()  # バッファをディスクに書き込む

        # 一時ファイルを開き、Whisperに渡す
        try:
            with open(temp_file.name, "rb") as f:
                result = openai.Audio.transcribe("whisper-1", f)
                full_transcript += result["text"] + "\n"
        finally:
            # 一時ファイルを削除
            os.remove(temp_file.name)

    return full_transcript


def split_file(audio_file: UploadedFile, chunk_size: int) -> List[BytesIO]:
    """
    Divide an audio file into chunks.
    """
    file_size = audio_file.size
    audio_data = audio_file.read()

    chunks = [
        BytesIO(audio_data[i : i + chunk_size])
        for i in range(0, len(audio_data), chunk_size)
    ]

    return chunks
