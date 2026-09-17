# -*- coding: utf-8 -*-
"""
@Time : 2026/9/16 15:43
@Author: janic
@File: rag_self_rag.py
"""
import warnings

from langchain_chroma import Chroma
# from langchain_classic.chains.summarize.refine_prompts import prompt_template
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, TypedDict

from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph

from model import QwenLLM, RagEmbedding
from rag_pipline import chroma_client, run_rag_pipline, query2doc
from rag_sub_query import question_rewrite

warnings.filterwarnings('ignore')

langchain_llm = QwenLLM()
embedding_model = RagEmbedding()


zhidu_db = Chroma("zhidu_db",
                  embedding_model.get_embedding_fun(),
                  client=chroma_client)
query = "公司的假有哪些，规定是什么？"
prompt_template = "你是企业员工助手，熟悉公司考勤和报销标准等规章制度。请根据下面的文档:\n\n{doc}\n 做一个简短的概括摘要改写，字数50字，并给出关键词"


def rag_retrieve(question, k=3):
    related_docs = zhidu_db.similarity_search(question, k=3)
    context = "\n".join([f"上下文{i+1}: {doc.page_content} \n" for i, doc in enumerate(related_docs)])
    return related_docs, context


def retrieve(state):
    print("------retrieve--------")
    state_dict = state["keys"]
    question = state_dict["question"]
    context_query = state_dict.get("context_query", None)
    query2doc_count = state_dict.get("query2doc_count", 0)
    rewrite_count = state_dict.get("rewrite_count", 0)

    if context_query is not None:
        documents, context = rag_retrieve(context_query, k=3)
    else:
        documents, context = rag_retrieve(question, k=3)

    return {"keys": {"context": context,
                     "question": question,
                     "documents": documents,
                     "query2doc_count": query2doc_count,
                     "rewrite_count": rewrite_count}}


def transform_query2doc(state):
    print("-------transform_query2doc--------")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    context = state_dict['context']
    query2doc_count = state_dict.get("query2doc_count", 0)
    rewrite_count = state_dict.get("rewrite_count", 0)

    context_query = query2doc(question)
    return {"keys": {"context": context,
                     "documents": documents,
                     "question": question,
                     "context_query": context_query,
                     "query2doc_count": query2doc_count+1,
                     "rewrite_count": rewrite_count}}


def transform_query_rewrite(state):
    print("-------transform_query_rewrite--------")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    context = state_dict['context']
    query2doc_count = state_dict.get("query2doc_count", 0)
    rewrite_count = state_dict.get("rewrite_count", 0)

    context_query = question_rewrite(question)

    return {"keys": {"context": context,
                     "documents": documents,
                     "question": question,
                     "context_query": context_query,
                     "query2doc_count": query2doc_count,
                     "rewrite_count": rewrite_count+1}}


def generate(state):
    print("-------generate--------")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    query2doc_count = state_dict.get("query2doc_count", 0)
    rewrite_count = state_dict.get("rewrite_count", 0)

    context = "\n".join([f"上下文{i+1}: {doc.page_content} \n" for i, doc in enumerate(documents)])
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template
    )

    llm_prompt = prompt.format(question=query, context=context)

    llm = langchain_llm

    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"context": context, "question": question})
    return {
        "keys": {"context": context,
                 "question": question,
                 "documents": documents,
                 "generation": generation,
                 "query2doc_count": query2doc_count,
                 "rewrite_count": rewrite_count},
    }


context_grade_prompt = PromptTemplate(
    template="""您是一名评分员，针对公司规章的问题，负责评估检索到的文档与用户问题的相关性。\n 
    以下是检索到的文档: \n\n {context} \n\n
    以下是用户问题: {question} \n
    如果文档包含与用户问题相关的关键词或语义信息，请将其评为相关. \n
    请给出一个二元分类 'Yes' 或 'No'，以指示该文档是否与问题相关.""",
    input_variables=["context", "question"],
)

context_grade_chain = context_grade_prompt | langchain_llm | StrOutputParser()

answer_useful_prompt = PromptTemplate(
                        template="""您是一名评分员，针对公司规章的问题，负责评估一个答案是否对解决用户问题有用。\n 
                        以下是答案: \n\n {generation} \n\n
                        以下是用户问题: {question} \n
                        请给出一个二元分类 'Yes' 或 'No'，以指示答案是否对解决问题有用.""",
                        input_variables=["generation", "question"],
                    )
answer_useful_chain = answer_useful_prompt | langchain_llm | StrOutputParser()

answer_supported_prompt = PromptTemplate(
                        template="""您是一名评分员，针对公司规章的问题，负责评估一个答案是否基于或者由一组事实支持。\n 
                        以下是事实: \n\n {context} \n\n
                        以下是答案: {generation} \n
                        请给出一个二元分类 'Yes' 或 'No'，以指示答案是否基于或者由一组事实支持.\n
                        不需要前言或解释""",
                        input_variables=["context", "generation"],
                    )
answer_supported_chain = answer_supported_prompt | langchain_llm | StrOutputParser()


def grade_documents(state):
    print("---Determines whether the retrieved documents are relevant to the question---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    context = state_dict["context"]
    query2doc_count = state_dict.get("query2doc_count", 0)
    rewrite_count = state_dict.get("rewrite_count", 0)

    # Score
    filtered_docs = []
    retrieve_enhance = "No"
    for d in documents:
        grade = context_grade_chain.invoke({"question": question, "context": d.page_content})
        if "yes" in grade.lower():
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            retrieve_enhance = "Yes"
            continue
    if query2doc_count > 0:
        retrieve_enhance = "No"
    context = ""
    if len(filtered_docs):
        context = "\n".join([f"上下文{i + 1}: {doc.page_content} \n" \
                             for i, doc in enumerate(filtered_docs)])
    return {
        "keys": {
            "context": context,
            "documents": filtered_docs,
            "question": question,
            "run_retrieve_enhance": retrieve_enhance,
            "query2doc_count": query2doc_count,
            "rewrite_count": rewrite_count
        }
    }


def grade_generation_v_documents_and_question(state):
    print("---Determines whether the answer is relevant to the question---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    context = state_dict["context"]
    query2doc_count = state_dict.get("query2doc_count", 0)
    generation = state_dict["generation"]
    rewrite_count = state_dict.get("rewrite_count", 0)

    grade = answer_supported_chain.invoke({"generation": generation, "context": context})

    if "yes" in grade.lower():
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_useful_chain.invoke({"question": question, "generation": generation})
        if "yes" in score.lower():
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            if rewrite_count < 1:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                return "end"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def decide_to_generate(state):

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    retrieve_enhance = state_dict["run_retrieve_enhance"]
    query2doc_count = state_dict.get("query2doc_count", 0)

    if retrieve_enhance == "Yes":
        if query2doc_count > 1:
            return "generate"
        else:
            print("---DECISION:  transform_query2doc---")
            return "transform_query2doc"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


class GraphState(TypedDict):
    keys: Dict[str, any]


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query2doc", transform_query2doc) # query2doc
workflow.add_node("transform_query_rewrite", transform_query_rewrite) # query2doc

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query2doc": "transform_query2doc",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query2doc", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": END,
        "useful": END,
        "end": END,
        "not useful": "transform_query_rewrite",
    },
)

workflow.add_edge("transform_query_rewrite", "retrieve")

app = workflow.compile()

query = "那个，我们公司有什么规定来着？"
inputs = {"keys": {"question": query}}

for output in app.stream(inputs):
    for key, value in output.items():
        pass

print(output['generate']['keys']['generation'])