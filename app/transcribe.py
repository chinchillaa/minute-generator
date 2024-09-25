import openai


def transcribe_audio(audio_file, open_ai_api_key: str) -> str:
    """
    Transcribe an audio file using the Whisper model in OpenAI's API.
    """
    openai.api_key = open_ai_api_key
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    return transcript["text"]
