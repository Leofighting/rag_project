# -*- coding: utf-8 -*-
"""
@Time : 2026/9/15 11:46
@Author: janic
@File: rag_hyde.py
"""
from langchain_classic.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain_core.prompts import PromptTemplate
import numpy as np

from model import QwenLLM, RagEmbedding

from rag_pipline import run_rag_pipline

langchain_llm = QwenLLM()
embedding_model = RagEmbedding()


def hyde(query, include_query=True):
    prompt_template = """你是一名公司员工制度的问答助手, 熟悉公司规章制度，请简短回答以下问题:
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(input_variables=['question'], template=prompt_template)
    llm_chain = LLMChain(llm=langchain_llm, prompt=prompt)
    # embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain,
    #                                           base_embeddings=embedding_model.get_embedding_fun())
    # hyde_embeddings = embeddings.embed_query(query)
    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=llm_chain,
        base_embeddings=embedding_model.get_embedding_fun(),
        prompt_key="web_search"
    )
    if include_query:
        query_embeddings = embedding_model.get_embedding_fun().embed_query(query)
        result = (np.array(query_embeddings) + np.array(hyde_embeddings)) / 2
        result = list(result)
    else:
        result = hyde_embeddings
    result = list(map(float, result))
    return result


query = "那个，我们公司有什么规定来着？"
answer = run_rag_pipline(query, hyde(query), context_query_type='vector')
print(answer)