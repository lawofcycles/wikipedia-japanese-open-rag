from typing import AsyncGenerator

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Initialize the LLM Engine
def init_engine():
    engine_args = AsyncEngineArgs(model=model_id, dtype='bfloat16', disable_log_requests=True, disable_log_stats=True)
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


def get_prompt(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)
    input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]


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


async def run(
    message: str,
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
    prompt = get_prompt(message=message, chat_history=chat_history, system_prompt=system_prompt)

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