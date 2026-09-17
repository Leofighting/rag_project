import jieba
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

import pickle
import warnings

from model import QwenLLM, RagEmbedding
from rag_pipline import chroma_client, run_rag_pipline

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


def jieba_preprocessing_func(text: str):
    return list(jieba.cut(text))

bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_preprocessing_func)
bm25_retriever.k = 3
query = "我要请病假100天"
ret_docs = bm25_retriever.invoke(query)

# for doc in ret_docs:
#     print("#"*88)
#     print(doc.page_content)

embedding_retriever = zhidu_db.as_retriever(search_kwargs={"k": 3})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, embedding_retriever],weights=[0.3, 0.7]
)

query1 = "公司丧假有什么规定？"

related_docs = ensemble_retriever.invoke(query1)[:3]
# for doc in related_docs:
#     print("#"* 88)
#     print(doc.page_content)

res1 = run_rag_pipline(query1, related_docs, k=3, context_query_type="doc")
print(res1)