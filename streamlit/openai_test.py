import openai

api_key = "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"
openai.organization = "org-z94aqqeoVXOPqHAWq7l4PSOb"
openai.api_key = api_key

request = openai.ChatCompletion.create(
    # 사용 모델
    model="gpt-3.5-turbo",
    # 전달 메세지
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ],
)
print(request)
