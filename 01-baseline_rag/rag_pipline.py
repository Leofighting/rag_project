# -*- coding: utf-8 -*-
"""
@Time : 2026/9/15 10:36
@Author: janic
@File: rag_pipline.py
"""
import warnings

from langchain_core.prompts import PromptTemplate

from model import RagEmbedding, RagLLM, QwenLLM
from langchain_chroma import Chroma
import chromadb
import numpy as np

warnings.filterwarnings("ignore")

embedding_model = RagEmbedding()
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
zhidu_db = Chroma("zhidu_db",
                  embedding_model.get_embedding_fun(),
                  client=chroma_client)

prompt_template = """
你是企业员工助手，熟悉公司考勤和报销标准等规章制度，需要根据提供的上下文信息context来回答员工的提问。\
请直接回答问题，如果上下文信息context没有和问题相关的信息，请直接先回答不知道 \
问题：{question} 
"{context}"
回答：
"""

llm = RagLLM()


def run_rag_pipline(query, context_query, k=3, context_query_type="query",
                    stream=True, prompt_template=prompt_template,
                    temperature=0.1):
    if context_query_type == "vector":
        related_docs = zhidu_db.similarity_search_by_vector(context_query, k=k)
    elif context_query_type == "query":
        related_docs = zhidu_db.similarity_search(context_query, k=k)
    elif context_query_type == "doc":
        related_docs = context_query
    else:
        related_docs = zhidu_db.similarity_search(context_query, k=k)
    context = "\n".join([f"上下文{i + 1}: {doc.page_content} \n" \
                         for i, doc in enumerate(related_docs)])
    print()
    print()
    print("#" * 100)
    print(f"query: {query}")
    print(f"context: {context}")
    # llm_prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template, )
    llm_prompt = prompt.format(question=query, context=context)

    if stream:
        response = llm(llm_prompt, stream=True)
        print(f"response: ")
        for chunk in response:
            print(chunk.choices[0].text, end='', flush=True)
        return ""
    else:
        response = llm(llm_prompt, stream=False, temperature=temperature)
        return response


def query2doc(query):
    prompt = f"你是一名公司员工制度的问答助手，熟悉公司规章制度，请简短回答以下问题：{query}"
    doc_info = llm(prompt, stream=False)
    context_query = f"{query}, {doc_info}"
    print("#"*88, "query2doc:")
    print(context_query)
    print("#"*88, "query2doc:")
    return context_query


query = "那个，我们公司有什么规定来着？"

# result1 = run_rag_pipline(query, query, k=3)
# print(result1)

# result2 = run_rag_pipline(query, query2doc(query), k=3)
# print(result2)

