from transcribe import transcribe_audio
from summarize import summarize_text

from io import BytesIO
import os

import math
import numpy as np
import pandas as pd

import torch
import openai
import whisper

import streamlit as st


class Config:
    OAI_API_KEY = os.environ["OPENAI_API_KEY"]
    SYSTEM_PROMPT = """
    以下の要件に基づき、文章を整理し議事録としてまとめてください。

    【要件】
    * 現状を把握し、まとめてください。
    * 上記に対する考えをまとめてください。
    * 今後取るべき具体的なアクションを記述してください。
    * 議事録以外の文章は記述しないでください。
    """
    CHUNK_SIZE = 25 * 1000 * 1000


def main():
    """
    Run the meeting transcription and summarization app.
    """
    st.title("Meeting Transcription and Summarization")

    audio_file = st.file_uploader(
        "Upload a meeting audio file", type=["mp3", "wav", "m4a"]
    )

    if audio_file is not None:
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(
                audio_file, Config.OAI_API_KEY, Config.CHUNK_SIZE
            )

        st.header("Transcript")
        st.text_area("", transcription, height=300)

        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(
                    transcription, Config.OAI_API_KEY, Config.SYSTEM_PROMPT
                )

            st.header("Meeting Summary")
            st.text_area("", summary, height=200)

            # 要約を.txtファイルとしてダウンロード可能にする
            txt_bytes = BytesIO(summary.encode("utf-8"))
            st.download_button(
                label="Download Summary as .txt",
                data=txt_bytes,
                file_name="meeting_summary.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
