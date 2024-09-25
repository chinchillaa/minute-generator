from transcribe import transcribe_audio
from summarize import summarize_text


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
            transcription = transcribe_audio(audio_file, Config.OAI_API_KEY)

        st.header("Transcript")
        st.text_area("", transcription, height=300)

        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(
                    transcription, Config.OAI_API_KEY, Config.SYSTEM_PROMPT
                )

            st.header("Meeting Summary")
            st.text_area("", summary, height=200)


if __name__ == "__main__":
    main()
