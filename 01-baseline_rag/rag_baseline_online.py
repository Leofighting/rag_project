# -*- coding: utf-8 -*-
"""
@Time : 2026/9/13 21:43
@Author: janic
@File: rag_baseline_online.py
"""
from model import RagEmbedding, RagLLM
from doc_parse import chunk, read_and_process_excel, logger

import pandas as pd
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

pdf_files = [r'D:\code_project\rag_project\01-baseline_rag\data\zhidu_employee.pdf',
             r'D:\code_project\rag_project\01-baseline_rag\data\zhidu_travel.pdf']
excel_files = [r'D:\code_project\rag_project\01-baseline_rag\data\zhidu_detail.xlsx']

r_spliter = RecursiveCharacterTextSplitter(chunk_size=128,
                                           chunk_overlap=30,
                                           separators=["\n\n",
                                                       "\n",
                                                       ".",
                                                       "\uff0e",
                                                       "\u3002",
                                                       ",",
                                                       "\uff0c",
                                                       "\u3001"
                                                       ])

doc_data = []
for pdf_file_name in pdf_files:
    res = chunk(pdf_file_name, callback=logger)
    for data in res:
        content = data["content_with_weight"]
        if "<table>" not in content and len(content) > 200:
            doc_data = doc_data + r_spliter.split_text(content)
        else:
            doc_data.append(content)


if __name__ == '__main__':
    # for i in doc_data:
    #     print(len(i), "=" * 11, i)
    for excel_file_name in excel_files:
        data = read_and_process_excel(excel_file_name)
        df = pd.DataFrame(data[8:], columns=data[7])
        data_excel = df.drop(columns=df.columns[11:17])
        doc_data.append(data_excel.to_markdown(index=False).replace(" ", ""))

    from langchain_core.documents import Document
    documents = []
    for chunk in doc_data:
        document = Document(
            page_content=chunk,
            metadata={"source": "test"},
        )
        documents.append(document)

    embedding_cls = RagEmbedding()

    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    embedding_db = Chroma.from_documents(documents,
                                         embedding_cls.get_embedding_fun(),
                                         client=chroma_client,
                                         collection_name="zhidu_db")