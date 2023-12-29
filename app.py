from typing import AsyncGenerator
import asyncio
import torch
import faiss
from time import time
import gradio as gr
from datasets.download import DownloadManager
from sentence_transformers import SentenceTransformer
from model_vllm import run
from datasets import load_dataset 

DEFAULT_SYSTEM_PROMPT = 'あなたは誠実で優秀な日本人のアシスタントです。'
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 512
MAX_INPUT_TOKEN_LENGTH = 4000

WIKIPEDIA_JA_DS = "singletongue/wikipedia-utils"
WIKIPEDIA_JS_DS_NAME = "passages-c400-jawiki-20230403"
WIKIPEDIA_JA_EMB_DS = "hotchpotch/wikipedia-passages-jawiki-embeddings"

def get_model(name: str, max_seq_length=512):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = max_seq_length
    return model

emb_model = get_model(name = "intfloat/multilingual-e5-large")

def get_wikija_ds(name: str = WIKIPEDIA_JS_DS_NAME):
    ds = load_dataset(path=WIKIPEDIA_JA_DS, name=name, split="train")
    return ds

ds = get_wikija_ds()

TITLE = '# ELYZA-japanese-Llama-2-13b-instruct'

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


def get_faiss_index(
    index_name: str, ja_emb_ds: str = WIKIPEDIA_JA_EMB_DS, name=WIKIPEDIA_JS_DS_NAME
):
    target_path = f"faiss_indexes/{name}/{index_name}"
    dm = DownloadManager()
    index_local_path = dm.download(
        f"https://huggingface.co/datasets/{ja_emb_ds}/resolve/main/{target_path}"
    )
    index = faiss.read_index(index_local_path)
    index.nprobe = 128
    return index

def text_to_emb(model, text: str, prefix: str):
    return model.encode([prefix + text], normalize_embeddings=True)

def search(
    faiss_index, emb_model, ds, question: str, search_text_prefix: str, top_k: int
):
    start_time = time()
    emb = text_to_emb(emb_model, question, search_text_prefix)
    emb_exec_time = time() - start_time
    scores, indexes = faiss_index.search(emb, top_k)
    faiss_seartch_time = time() - emb_exec_time - start_time
    scores = scores[0]
    indexes = indexes[0]
    results = []
    for idx, score in zip(indexes, scores):  # type: ignore
        idx = int(idx)
        passage = ds[idx]
        results.append((score, passage))
    return results, emb_exec_time, faiss_seartch_time

async def generate(
    question: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
) -> AsyncGenerator[list[tuple[str, str]], None]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    global emb_model
    global ds
    emb_model_pq = "256"
    index_emb_model_name = "multilingual-e5-large-passage"
    index_name = f"{index_emb_model_name}/index_IVF2048_PQ{emb_model_pq}.faiss"
    faiss_index = get_faiss_index(index_name=index_name)
    faiss_index.nprobe = 128

    contexts = []
    scores = []
    search_results, emb_exec_time, faiss_seartch_time = search(
        faiss_index,
        emb_model,
        ds,
        question,
        search_text_prefix="passage",
        top_k=3,
    )
    for score, passage in search_results:
        scores.append(score)
        contexts.append(passage)

    history = history_with_input[:-1]
    stream = await run(
        question=question,
        contexts = contexts,
        chat_history=history,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=float(repetition_penalty),
        stream=True,
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
            height=600,
        )
        with gr.Column():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder='指示を入力してください。例: カレーとハンバーグを組み合わせた美味しい料理を3つ教えて',
                scale=10,
                lines=10,
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
        system_prompt = gr.Textbox(label='システムプロンプト', value=DEFAULT_SYSTEM_PROMPT, lines=8)
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
            system_prompt,
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
                system_prompt,
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
            system_prompt,
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