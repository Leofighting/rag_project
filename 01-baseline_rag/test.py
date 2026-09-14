# -*- coding: utf-8 -*-
"""
@Time : 2026/9/14 10:19
@Author: janic
@File: test.py
"""
from model import RagEmbedding, RagLLM
from doc_parse import chunk, read_and_process_excel, logger

import pandas as pd
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

llm = RagLLM()
