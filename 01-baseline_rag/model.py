# -*- coding: utf-8 -*-
"""
@Time : 2026/9/13 16:39
@Author: janic
@File: model.py
"""
import warnings
from typing import Any, List, Optional

from openai import OpenAI
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")


class RagLLM(object):
    client: Optional[Any] = None

    def __init__(self):
        super().__init__()
        self.client = OpenAI(base_url="http://localhost:11434/v1/",
                             api_key="qwen3:14b")

    def __call__(self, prompt: str, **kwargs):
        completion = self.client.completions.create(model="qwen3:14b",
                                                   prompt=prompt,
                                                   temperature=kwargs.get("temperature", 0.1),
                                                   top_p=kwargs.get("top_p", 0.9),
                                                   max_tokens=kwargs.get("max_tokens", 4096),
                                                   stream=kwargs.get("stream", False))
        if kwargs.get("stream", False):
            return completion
        return completion.choices[0].text


class QwenLLM(LLM):
    client: Optional[Any] = None

    def __init__(self):
        super().__init__()
        self.client = OpenAI(base_url="http://localhost:11434/v1/",
                             api_key="qwen3:14b")

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_prompt: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        completion = self.client.completions.create(model="qwen3:14b",
                                                    prompt=prompt,
                                                    temperature=kwargs.get("temperature", 0.1),
                                                    top_p=kwargs.get("top_p", 0.9),
                                                    max_tokens=kwargs.get("max_tokens", 4096),
                                                    stream=kwargs.get("stream", False))
        return completion.choices[0].text

    # def _call(self,
    #           prompt: str,
    #           stop: Optional[List[str]] = None,
    #           run_manager: Optional[CallbackManagerForLLMRun] = None,
    #           **kwargs: Any):
    #     completion = self.client.chat.completions.create(
    #         model="qwen3:14b",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=kwargs.get("temperature", 0.1),
    #         top_p=kwargs.get("top_p", 0.9),
    #         max_tokens=kwargs.get("max_tokens", 4096),
    #         response_format={"type": "json_object"},  # 关键：强制 JSON 输出
    #         stream=kwargs.get("stream", False)
    #     )
    #     return completion.choices[0].message.content

    @property
    def _llm_type(self)->str:
        return "rag_llm_qwen3:14b"


class RagEmbedding(object):
    def __init__(self, model_path=r"D:\code_project\models\bge-m3",
                 device="auto"):
        self.embedding = HuggingFaceEmbeddings(model_name=model_path,
                                               model_kwargs={"device": "cuda"})

    def get_embedding_fun(self):
        return self.embedding


