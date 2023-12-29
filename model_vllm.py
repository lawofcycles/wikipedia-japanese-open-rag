from typing import AsyncGenerator

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

DEFAULT_QA_PROMPT = """
## Instruction

参考情報を元に、質問への回答や洞察、コメントをしてください。
回答は参考情報だけを元に作成し、推測を含めないでください。参考情報に答えが見つからなかった場合は、その旨を述べてください。

## 参考情報

{contexts}

## 質問

{question}
""".strip()

# Initialize the LLM Engine
def init_engine():
    engine_args = AsyncEngineArgs(model=model_id, dtype='bfloat16',tensor_parallel_size=4, disable_log_requests=True, disable_log_stats=True,gpu_memory_utilization=0.6)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    return engine


model_id = 'elyza/ELYZA-japanese-Llama-2-13b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
engine = init_engine()


# Generator function for streaming response
async def stream_results(prompt, sampling_params):
    global engine
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    async for request_output in results_generator:
        text_outputs = [output.text for output in request_output.outputs]
        yield text_outputs


def get_prompt(question: str, contexts: str,  chat_history: list[tuple[str, str]], system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        
    message = DEFAULT_QA_PROMPT.format(contexts=to_contexts(contexts), question=question)
    
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


# Function to generate a response
async def generate_response(engine, prompt: str):
    request_id = random_uuid()
    sampling_params = SamplingParams()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    text_outputs = [output.text for output in final_output.outputs]
    return text_outputs

def to_contexts(passages):
    contexts = ""
    for passage in passages:
        title = passage["title"]
        text = passage["text"]
        # section = passage["section"]
        contexts += f"- {title}: {text}\n"
    return contexts

async def run(
    question: str,
    contexts: list[str],
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = False,
    repetition_penalty: float = 1.2,
    stream: bool = False,
) -> AsyncGenerator | str:
    request_id = random_uuid()

    prompt = get_prompt(question=question, contexts = contexts,chat_history=chat_history, system_prompt=system_prompt)

    if not do_sample:
        # greedy
        temperature = 0
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    results_generator = engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    )

    async def stream_results() -> AsyncGenerator:
        async for request_output in results_generator:
            yield ''.join([output.text for output in request_output.outputs])

    if stream:
        return stream_results()
    else:
        async for request_output in results_generator:
            pass
        return ''.join([output.text for output in request_output.outputs])