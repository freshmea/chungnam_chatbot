import gradio as gr


def answer(state, state_chatbot, text):
    # msg 를 챗봇 api 로 전달해서 할당 받는 코드가 필요함.
    # chat gpt api 가 필요함.
    msg = "test_msg"

    new_state = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": msg},
    ]

    state = state + new_state
    state_chatbot = state_chatbot + [(text, msg)]

    print(state)

    return state, state_chatbot, state_chatbot


with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State([{"role": "system", "content": "You are a helpful assistant."}])
    state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML(
            """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>test-3.5</h1>
            </div>
        </div>"""
        )

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Send a message...").style(
            container=False
        )

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: "", None, txt)

demo.launch(debug=True, share=True)
