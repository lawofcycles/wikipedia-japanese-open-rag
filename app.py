from typing import AsyncGenerator, List, Tuple
import gradio as gr
import httpx

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 512
MAX_INPUT_TOKEN_LENGTH = 4000
API_URL = "http://localhost:8000/question"

TITLE = '## multilingual-e5-largeとELYZA-japanese-Llama-2-13b-instructによるWikipedia日本語ページをコーパスとするRAGアプリ'

def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message


def display_input(message: str, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history


def delete_prev_fn(history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''


async def rag_inf_api(
    question: str,
    history_with_input: list[tuple[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,) -> AsyncGenerator:

    data = {
    "question": question,
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "do_sample": do_sample,
    "repetition_penalty": repetition_penalty
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, json=data)

            if response.status_code != 200:
                raise Exception(f"Error: Server responded with status code {response.status_code}")

            last_response = ""
            async for line in response.aiter_lines():
                if line:
                    new_response = line.strip()
                    if new_response.startswith(last_response):
                        new_characters = new_response[len(last_response):]
                        last_response += new_response
                        print("contain: " + last_response)
                        if new_characters:
                            yield ''.join(last_response)
                    else:
                        last_response = new_response
                        print("not contain: " + last_response)
                        yield ''.join(last_response)
        except httpx.HTTPError as e:
            raise Exception(f"HTTP request failed: {str(e)}")


async def generate(
    question: str,
    history_with_input: list[tuple[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
) -> AsyncGenerator[list[tuple[str, str]], None]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    history = history_with_input[:-1]
    stream = rag_inf_api(
        question,
        history_with_input,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        do_sample,
        repetition_penalty,
    )
    async for response in stream:
        yield history + [(question, response)]


def convert_history_to_str(history: list[tuple[str, str]]) -> str:
    res = []
    for user_utt, sys_utt in history:
        res.append(f'😃: {user_utt}')
        res.append(f'🤖: {sys_utt}')
    return '<br>'.join(res)


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(TITLE)
    with gr.Group():
        chatbot = gr.Chatbot(
            label='Chatbot',
            height=400,
        )
        with gr.Column():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder='おとぎ銃士 赤ずきんのあらすじを詳しく教えて',
                scale=10,
                lines=3,
            )
            submit_button = gr.Button(
                '送信', variant='primary', scale=1, min_width=0
            )
    with gr.Row():
        retry_button = gr.Button('🔄  同じ入力でもう一度生成', variant='secondary')
        undo_button = gr.Button('↩️ ひとつ前の状態に戻る', variant='secondary')
        clear_button = gr.Button('🗑️  これまでの出力を消す', variant='secondary')

    saved_input = gr.State()
    uuid_list = gr.State([])

    with gr.Accordion(label='上の対話履歴をスクリーンショット用に整形', open=False):
        output_textbox = gr.Markdown()

    with gr.Accordion(label='詳細設定', open=False):
        max_new_tokens = gr.Slider(
            label='最大出力トークン数',
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )
        repetition_penalty = gr.Slider(
            label='Repetition penalty',
            minimum=1.0,
            maximum=10.0,
            step=0.1,
            value=1.0,
        )
        do_sample = gr.Checkbox(label='do_sample', value=False)
        temperature = gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=1.0,
        )
        top_p = gr.Slider(
            label='Top-p (nucleus sampling)',
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        )
        top_k = gr.Slider(
            label='Top-k',
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        )

    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).success(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            do_sample,
            repetition_penalty,
        ],
        outputs=chatbot,
        api_name=False,
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

    button_event_preprocess = (
        submit_button.click(
            fn=clear_and_save_textbox,
            inputs=textbox,
            outputs=[textbox, saved_input],
            api_name=False,
            queue=False,
        )
        .then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        ).success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                do_sample,
                repetition_penalty,
            ],
            outputs=chatbot,
            api_name=False,
        )
        .then(
            fn=convert_history_to_str,
            inputs=chatbot,
            outputs=output_textbox,
        )
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            do_sample,
            repetition_penalty,
        ],
        outputs=chatbot,
        api_name=False,
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=saved_input,
        outputs=textbox,
        api_name=False,
        queue=False,
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

    clear_button.click(
        fn=lambda: ([], ''),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

demo.queue(max_size=5).launch(server_name='0.0.0.0',share=True)
