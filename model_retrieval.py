import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

load_dotenv()
MILVUS_HOST = os.environ.get("MILVUS_HOST")
MILVUS_PORT = os.environ.get("MILVUS_PORT")
DB_NAME = os.environ.get("DB_NAME")
TOP_K = int(os.environ.get("TOP_K", 5))
DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", 0.6))
SPARSE_WEIGHT = float(os.environ.get("SPARSE_WEIGHT", 0.4))

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, db_name=DB_NAME)

# 💡 [핵심] 처음에 파일이 임포트될 때는 빈 껍데기만 만들어 둡니다. (VRAM 소모 0)
embed_model = None
rerank_model = None
global_collections = {}


def execute_hybrid_retrieval(query, collection_name):
    global embed_model, rerank_model, global_collections

    if embed_model is None:
        embed_model = BGEM3FlagModel(os.environ.get("EMBED_MODEL_ID", 'BAAI/bge-m3'), use_fp16=True)
    if rerank_model is None:
        rerank_model = CrossEncoder(os.environ.get("RERANKER_MODEL_ID"), trust_remote_code=True)

    # 1. DB 불러오기
    if collection_name not in global_collections:
        collection = Collection(collection_name)
        collection.load()
        global_collections[collection_name] = collection
    else:
        collection = global_collections[collection_name]

    # 3. 질의 임베딩 (Dense & Sparse 동시 추출)
    embeddings = embed_model.encode([query], return_dense=True, return_sparse=True)
    query_dense_vector = embeddings['dense_vecs'][0].tolist()
    query_sparse_vector = embeddings['lexical_weights'][0]

    # 4-1. Dense (의미) 검색 요청
    req_dense = AnnSearchRequest(
        data=[query_dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=40
    )

    # 4-2. Sparse (키워드/ES역할) 검색 요청
    req_sparse = AnnSearchRequest(
        data=[query_sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=40
    )

    # 4-3. 가중치 랭커 설정 (Dense 0.6 : Sparse 0.4)
    ranker = WeightedRanker(DENSE_WEIGHT, SPARSE_WEIGHT)

    # 5. 하이브리드 검색 실행
    results = collection.hybrid_search(
        reqs=[req_dense, req_sparse],
        rerank=ranker,
        limit=40,
        output_fields=["id", "type", "page", "path", "content", "source", "parent_id"]  # 🌟 7개 컬럼 명시
    )
    initial_hits = results[0]

    # 6. 리랭킹
    pairs = [[query, hit.entity.get("content")] for hit in initial_hits]
    # pairs = [[query, hit.entity.get("text")[:1500]] for hit in initial_hits]
    # scores = rerank_model.predict(pairs)
    # scores = rerank_model.predict(pairs, batch_size=4)
    scores = rerank_model.predict(pairs, batch_size=4, max_length=1024)
    unique_dict = {}
    for i, hit in enumerate(initial_hits):
        doc = {
            "score": float(scores[i]),
            # Rerank X
            # "score": float(hit.distance),  # Milvus 하이브리드가 계산한 순수 점수를 그대로 사용
            "page_content": hit.entity.get("content"),
            "metadata": {
                "id": hit.entity.get("id"),
                "type": hit.entity.get("type"),
                "page": hit.entity.get("page"),
                "path": hit.entity.get("path"),
                "source": hit.entity.get("source"),
                "parent_id": hit.entity.get("parent_id")
            }
        }
        text_key = doc["page_content"].strip()
        if text_key not in unique_dict:
            unique_dict[text_key] = doc
        else:
            if doc["metadata"]["parent_id"] and not unique_dict[text_key]["metadata"]["parent_id"]:
                unique_dict[text_key] = doc
    # 8. 최종 결과 정렬 및 반환
    retrieved_docs = sorted(unique_dict.values(), key=lambda x: x["score"], reverse=True)[:TOP_K]

    # 출력창이 너무 지저분해지는 걸 막으려고 이 프린트문은 필요시 주석 처리해도 좋습니다.
    # print(f"✨ [Hybrid Retrieval] 하이브리드 검색 완료! (Dense {DENSE_WEIGHT}:Sparse {SPARSE_WEIGHT}) 총 {len(retrieved_docs)}개의 청크를 찾았습니다.")

    return retrieved_docs