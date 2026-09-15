# -*- coding: utf-8 -*-
"""
@Time : 2026/9/15 15:33
@Author: janic
@File: rag_sub_query.py
"""
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from model import QwenLLM, RagEmbedding

from rag_pipline import run_rag_pipline

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


query3 = "我有事外出，要怎么办"
new_query = take_step_back(query3)
res3 = run_rag_pipline(new_query, new_query, k=3)
print(res3)