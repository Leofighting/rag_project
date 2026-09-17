# -*- coding: utf-8 -*-
"""
@Time : 2026/9/15 15:33
@Author: janic
@File: rag_sub_query.py
"""
from langchain_chroma import Chroma
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryByteStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from typing import List
# from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator


from model import QwenLLM, RagEmbedding

from rag_pipline import run_rag_pipline, chroma_client

langchain_llm = QwenLLM()
embedding_model = RagEmbedding()


def sub_question(query):
    prompt_template_ = """你是一名公司员工制度的问答助手, 熟悉公司规章制度。你的任务是对复杂问题继续拆解，以便以理解员工的意图，请根据以下问题创建一个子问题列表：

    复杂问题：{question}

    请执行以下步骤：

    识别主要主题：找出问题中的核心主题或概念。
    分解成子问题：将主要问题分解成可以独立理解和解决的多个子问题。

    子问题列表:"""
    prompt = PromptTemplate(input_variables=["question"],
                            template=prompt_template_)
    llm_chain = LLMChain(llm=langchain_llm, prompt=prompt)
    sub_query = llm_chain.invoke(query)['text'].split("\n")
    return sub_query


query1 = "最近发生了很多的事情，有点感冒发烧， 还要出差去上海，我可以请什么假？"
sub_queries = sub_question(query1)
# print(sub_queries)
#
# print("#"*88)
#
# for sub_query in sub_queries:
#     res = run_rag_pipline(sub_query, sub_query, k=3, context_query_type="query")
#     print(res)


def question_rewrite(query):
    prompt_template = """你是一名公司员工制度的问答助手, 熟悉公司规章制度。
        你的任务是需要为给定的问题，从不同层次生成这个问题的转述版本，使其更易于检索，转述的版本增加一些公司规章制度的关键词
        问题: {question}
        转述版本:"""
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    llm_chain = LLMChain(llm=langchain_llm, prompt=prompt)
    return llm_chain.invoke(query)['text']


# query2 = "我想了解下,临时外出需要怎么申请"
# print("##"*55)
# print("query_rewrite")
# rewrite_query = question_rewrite(query2)
# print(rewrite_query)
# res2 = run_rag_pipline(rewrite_query, rewrite_query, k=3)
# print(res2)


def take_step_back(query):
    examples = [
        {
            "input": "我祖父去世了，我要回去几天",
            "output": "公司丧葬假有什么规定？",
        },
        {
            "input": "我去北京出差，北京的消费高，有什么额外的补助",
            "output": "员工出差的交通费、住宿费、伙食补助费的规定是什么",
        },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}")
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一名公司员工制度的问答助手, 熟悉公司规章制度。你的任务是将输入的问题通过归纳、提炼，转换为关于公司规章制定相关的一般性的问题，使得这些问题更容易捕捉问题的意图。请参考下面的例子：""",
            ),
            few_shot_prompt,
            ("user", "{question}")
        ]
    )
    question_gen = prompt | langchain_llm | StrOutputParser()

    res = question_gen.invoke({"question": query})
    return res.split("<|endoftext|>")[0]


# query3 = "我有事外出，要怎么办"
# new_query = take_step_back(query3)
# res3 = run_rag_pipline(new_query, new_query, k=3)
# print(res3)


import pickle

with open("./data/zhidu_db.pickl", "rb") as file:
    doc_txts = pickle.load(file)

doc_ids = list(doc_txts.keys())
docs = list(doc_txts.values())
# print(docs[:4])

chile_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=64,
    chunk_overlap=15,
    separators=["\n\n",
              "\n",
              ".",
              "\uff0e",  # Fullwidth full stop
               "\u3002",  # Ideographic full stop
              ",",
              "\uff0c",  # Fullwidth comma
              "\u3001",  # Ideographic comma
             ])

sub_docs = []
id_key = "doc_id"
index_type = "sm_chunk"

for i, doc in enumerate(docs):
    _id = doc_ids[i]
    if doc.metadata["is_table"] == 1:
        _doc = Document(page_content=doc.page_content,
                        metadata={"type": index_type, id_key: _id})
        sub_docs.extend([_doc])
        continue

    _sub_docs = chile_text_splitter.split_documents([doc])

    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
        _doc.metadata["type"] = index_type

    sub_docs.extend(_sub_docs)

# print(sub_docs)

sm_chunk_db = Chroma.from_documents(sub_docs,
                                    embedding_model.get_embedding_fun(),
                                    client=chroma_client,
                                    collection_name="zhidu_db_sm_chunk")

store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=sm_chunk_db,
    byte_store=store,
    id_key=id_key
)

# retriever.docstore.mset(list(zip(doc_ids, docs)))

# query4 = "出差交通费怎么算？"
# res4 = retriever.invoke(query4)
# print(res4)
#
# answer4 = sm_chunk_db.similarity_search(query4, k=2)
# print(answer4)

# llm = QwenLLM()
# prompt_template = "你是企业员工助手，熟悉公司考勤和报销标准等规章制度。请根据下面的文档:\n\n{doc}\n 做一个简短的概括摘要改写，字数50字，并给出关键词"
#
# chain = (
#     {"doc": lambda x: x.page_content}
#     | ChatPromptTemplate.from_template(prompt_template)
#     | llm
#     | StrOutputParser()
# )
#
# summaries = chain.batch(docs, {"max_concurrency": 1})
#
# summary_docs = []
# id_key = "doc_id"
# index_type = "summary"
# for i, s in enumerate(summaries):
#     _id = doc_ids[i]
#     doc = docs[i]
#     if doc.metadata["is_table"] == 1:
#         _doc = Document(page_content=doc.page_content,
#                         metadata={"type": index_type, id_key: _id})
#         summary_docs.extend([_doc])
#         continue
#
#     _s = Document(page_content=s,
#                   metadata={"type": index_type, id_key: _id})
#
#     summary_docs.extend([_s])

# summary_chunk_db = Chroma.from_documents(summary_docs,
#                                          embedding_model.get_embedding_fun(),
#                                          client=chroma_client,
#                                          collection_name="zhidu_db_summary")
# store = InMemoryByteStore()
# id_key = "doc_id"
# summary_retriever = MultiVectorRetriever(
#     vectorstore=summary_chunk_db,
#     byte_store=store,
#     id_key=id_key
# )
# summary_retriever.docstore.mset(list(zip(doc_ids, docs)))
#
# query5 = "出差交通费怎么算"
# res5 = summary_retriever.invoke(query5)
# print(res5)

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template(
    "你是企业员工助手，熟悉公司考勤和报销标准等规章制度。你的任务是提出在下面文档的内容中可以找到答案的3个假设性问题。:\n\n{doc}, 要求输出为中文，不包含解释性内容，格式列表格式如['问题1'， '问题2', '问题3'] "
    )
    | llm
)

# chain.invoke(docs[4])


class HypotheticalQuestions(BaseModel):
    questions: List[str] = Field(..., description="List of questions")


question_docs = []
id_key = "doc_id"
index_type = "hq"
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    for _ in range(3):
        try:
            hq = chain.invoke(doc)
            res = eval(hq)
            q = HypotheticalQuestions(questions=res)
            for i, question in enumerate(q.questions):
                question_docs.extend(
                    [Document(page_content=question, metadata={"type": index_type, id_key:_id})]
                )
            break
        except:
            continue


hq_chunk_db = Chroma.from_documents(question_docs,
                                    embedding_model.get_embedding_fun(),
                                    client=chroma_client,
                                    collection_name="zhidu_db_hq")

store = InMemoryByteStore()
id_key = "doc_id"

hq_retriever = MultiVectorRetriever(
    vectorstore=hq_chunk_db,
    byte_store=store,
    id_key=id_key
)

hq_retriever.docstore.mset(list(zip(doc_ids, docs)))
query = "出差交通费怎么算？"
res6 = hq_retriever.invoke(query)
# print(res6)

answer6 = hq_chunk_db.similarity_search(query, k=2)
print(answer6)