from typing import List, Tuple, AsyncGenerator
import torch
from transformers import AutoTokenizer
from search import create_text_searcher
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

DEFAULT_SYSTEM_PROMPT = 'あなたは誠実で優秀な日本人のアシスタントです。'
DEFAULT_QA_PROMPT = """
## Instruction

参考情報を元に、質問に回答してください。
回答は参考情報だけを元に作成し、推測や一般的な知識を含めないでください。参考情報に答えが見つからなかった場合は、その旨を述べてください。

## 参考情報

{contexts}

## 質問

{question}
""".strip()

LLM_MODEL_ID = 'elyza/ELYZA-japanese-Llama-2-13b-instruct'


class InferenceEngine:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise EnvironmentError('need CUDA env.')
        self.llm_engine = self.init_llm_engine()
        self.searcher = create_text_searcher()
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    def init_llm_engine(self) -> AsyncLLMEngine:
        engine_args = AsyncEngineArgs(model=LLM_MODEL_ID, dtype='bfloat16',
                                      tensor_parallel_size=4,
                                      disable_log_requests=True,
                                      disable_log_stats=True,
                                      gpu_memory_utilization=0.6)
        return AsyncLLMEngine.from_engine_args(engine_args)

    def get_prompt(self, question: str, contexts: List[str], system_prompt: str) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        message = DEFAULT_QA_PROMPT.format(contexts=self.searcher.to_contexts(contexts), question=question)
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    async def generate_response(self, prompt: str, sampling_params: SamplingParams) -> List[str]:
        request_id = random_uuid()
        results_generator = self.llm_engine.generate(prompt, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        assert final_output is not None
        text_outputs = [output.text for output in final_output.outputs]
        return text_outputs

    async def run(
        self,
        question: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = False,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        contexts, scores = await self.search_contexts(question)
        prompt = self.get_prompt(question, contexts, system_prompt)

        if not do_sample:
            temperature = 0
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        results = await self.generate_response(prompt, sampling_params)
        return results

    async def search_contexts(self, question: str) -> Tuple[List[str], List[float]]:
        search_results, emb_exec_time, faiss_search_time = self.searcher.search(
            question,
            top_k=5,
        )
        scores, contexts = zip(*search_results)
        return contexts, scores