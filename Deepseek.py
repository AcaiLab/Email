from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

MODEL_NAME = "qwen3-8b-deepseek-v3.2-speciale-distill"


def query_deepseek(system_prompt, user_prompt):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=200
    )

    return response.choices[0].message.content