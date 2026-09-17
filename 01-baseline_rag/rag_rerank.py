# -*- coding: utf-8 -*-
"""
@Time : 2026/9/16 12:00
@Author: janic
@File: rag_rerank.py
"""
import os
import pickle
import warnings

import numpy as np
import torch

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model import QwenLLM, RagEmbedding
from rag_pipline import chroma_client, run_rag_pipline

from rag_maxin_query import jieba_preprocessing_func

warnings.filterwarnings('ignore')

langchain_llm = QwenLLM()
embedding_model = RagEmbedding()

with open("./data/zhidu_db.pickl", "rb") as file:
    doc_txts = pickle.load(file)

zhidu_db = Chroma("zhidu_db",
                  embedding_model.get_embedding_fun(),
                  client=chroma_client)

doc_ids = list(doc_txts.keys())
docs = list(doc_txts.values())

model_path = r"D:\code_project\models\bge-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
rerank_model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()

bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_preprocessing_func)

query = "我要请病假100天"
ret_docs = bm25_retriever.invoke(query)

pairs = []
for doc in ret_docs:
    pairs.append([query, doc.page_content])

inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)

with torch.no_grad():
    inputs = {key: inputs[key].cuda() for key in inputs.keys()}
    scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

doc_sort_ids = list((scores.cpu().numpy()*-1).argsort())
# print(doc_sort_ids)

adocs = [np.array(ret_docs)[doc_sort_ids][0]]
answer1 = run_rag_pipline(query, adocs, k=3, context_query_type="doc")
print(answer1)