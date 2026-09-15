# -*- coding: utf-8 -*-
"""
@Time : 2026/9/14 10:19
@Author: janic
@File: test.py
"""
import warnings
warnings.filterwarnings('ignore')
from model import RagEmbedding, RagLLM, QwenLLM
from langchain_chroma import Chroma
import chromadb

llm = RagLLM()

prompt_template = """
你是企业员工助手，熟悉公司考勤和报销标准等规章制度，需要根据提供的上下文信息context来回答员工的提问。\
请直接回答问题，如果上下文信息context没有和问题相关的信息，请直接回答[不知道,请咨询HR] \
问题：{question} 
"{context}"
回答：
"""

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
embedding_model = RagEmbedding()
zhidu_db = Chroma("zhidu_db",
                  embedding_model.get_embedding_fun(),
                  client=chroma_client)


def run_rag_pipline_without_stream(query, k=3):
    related_docs = zhidu_db.similarity_search(query, k=k)
    context_list = [f"上下文{i+1}: {doc.page_content} \n" for i, doc in enumerate(related_docs)]
    context = "\n".join(context_list)
    llm_prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    response = llm(llm_prompt, stream=False)
    return response, context_list


questions = [
    "伙食补助费标准是什么?",
    "出差可以买意外保险吗？需要自己购买吗",
]
ground_truths = [
    "伙食补助费标准: 西藏、青海、新疆 120元/人、天 其他省份 100元/人、天",
    "出差可以购买交通意外保险，由单位统一购买，不再重复购买",
]

answers = []
contexts = []

for query in questions:
    response, context_list = run_rag_pipline_without_stream(query, k=3)
    answers.append(response)
    contexts.append(context_list)

from datasets import Dataset

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(data)

from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas import evaluate, RunConfig

eval_llm = QwenLLM()
embedding_model = RagEmbedding()
eval_embedding_fn = embedding_model.get_embedding_fun()
config = RunConfig(timeout=1200, log_tenacity=True)

result = evaluate(
    dataset=dataset,
    llm=eval_llm,
    embeddings=eval_embedding_fn,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
    raise_exceptions=True,
    run_config=config
)

df = result.to_pandas()

print(df)