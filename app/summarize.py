import openai


def summarize_text(
    transcription_text: str, open_ai_api_key: str, system_prompt: str
) -> str:
    """
    Summarize the given text using the Chat Completion API in OpenAI.
    """
    openai.api_key = open_ai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription_text},
        ],
    )
    summary = response["choices"][0]["message"]["content"]

    return summary
