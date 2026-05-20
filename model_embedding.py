import os, time
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel


def generate_embeddings(chunk_list):
    """랭체인 없이 순정 sentence_transformers를 사용한 임베딩 생성"""
    start_time = time.time()
    load_dotenv()
    EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HF_TOKEN", "")
    model = BGEM3FlagModel(EMBED_MODEL_ID, use_fp16=True)

    # 텍스트 추출 (Parent-Child, Fixed-Length 모두 대응)
    contents = [chunk.get("content", "") for chunk in chunk_list]

    # 모델 준비 (랭체인 래퍼 없이 직접 로드)
    embeddings = model.encode(contents, return_dense=True, return_sparse=True, return_colbert_vecs=False)

    # 벡터 변환 (순정 모델은 encode()를 사용하며, Milvus에 넣기 위해 tolist()로 변환)
    dense_vectors = embeddings['dense_vecs'].tolist()
    sparse_vectors = embeddings['lexical_weights']

    print(f"--- [Step 6] 임베딩 완료! (Dense 벡터 차원: {len(dense_vectors[0])}, Sparse 벡터 추출 성공, {time.time() - start_time:.2f}초 소요 ---)")
    return dense_vectors, sparse_vectors