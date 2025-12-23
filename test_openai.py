from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o-mini",
    input="こんにちは。短く自己紹介して。"
)

print(resp.output_text)
