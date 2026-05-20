# Milvus
import os, time
from dotenv import load_dotenv
from pymilvus import connections, db, Collection, FieldSchema, CollectionSchema, DataType, utility

# Fixed-length 및 Parent-Child Chunking
def milvus_unified_insert(chunk_list, dense_vectors, sparse_vectors, collection_name, doc_filename):
    """
    FL, PC 방식 상관없이 모든 메타데이터를 JSON 필드에 담아 적재하는 통합 함수
    """
    start_time = time.time()
    # 1. 서버 연결 및 DB 생성
    load_dotenv()
    MILVUS_HOST = os.environ.get("MILVUS_HOST")  # "127.0.0.1"
    MILVUS_PORT = os.environ.get("MILVUS_PORT")  # "19530"
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    if "edu" not in db.list_database():
        db.create_database("edu")
    db.using_database("edu")

    # 2. 스키마 설계
    fields = [
        # PK 및 벡터 필드
        FieldSchema(name="milvus_id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Milvus 내부 고유 키
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=len(dense_vectors[0])),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        #  7개 컬럼 매핑
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=500),  # Docling ID / 청크 고유 ID
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
        # chunk_fl, chunk_pc, table_text, picture, table_image
        FieldSchema(name="page", dtype=DataType.INT64, is_nullable=True),  # 페이지 번호 (정수형으로 일치시켜 검색/필터링 최적화)
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1000, is_nullable=True),  # 이미지 저장 경로 (텍스트는 None)
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 실제 검색 및 LLM 컨텍스트용 보강 텍스트 본문
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500, partition_key=True),  # 파티션 키로 활용할 원본 파일명
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=1000, is_nullable=True)  # 부모 계층 제목 구조 정보
    ]

    # 혹시 같은 이름의 db가 있으면 제거 후 새로 만들기
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    schema = CollectionSchema(fields, description=f"Hybrid RAG collection for {doc_filename}")
    collection = Collection(name=collection_name, schema=schema)
    # 3. 데이터 적재 (스키마 순서에 완벽히 맞춰야 함)
    # milvus_id는 auto_id이므로 데이터를 넣지 않습니다. dense_vector부터 순서대로 준비합니다.

    # None -> 숫자일 때 0, 문자일 때 ""
    ids = [chunk.get("id", "") for chunk in chunk_list]
    types = [chunk.get("type", "") for chunk in chunk_list]
    pages = [chunk.get("page") if chunk.get("page") is not None else 0 for chunk in chunk_list]  # 정수형 숫자, None -> 0으로 변환
    paths = [chunk.get("path") if chunk.get("path") is not None else "" for chunk in chunk_list]  # 경로 문자열, None -> ""으로변환
    contents = [chunk.get("content", "") for chunk in chunk_list]  # 보강된 최종 텍스트 본문
    sources = [chunk.get("source", doc_filename) for chunk in chunk_list]  # 통일된 doc_filename 반영
    parent_ids = [chunk.get("parent_id") if chunk.get("parent_id") is not None else "" for chunk in chunk_list]  # 부모 구조, None -> ""으로변환

    # Milvus에 벌크로 찌르는 데이터 테이블 구조화
    milvus_batch_data = [
        dense_vectors,  # 1. dense_vector
        sparse_vectors,  # 2. sparse_vector
        ids,  # 3. id
        types,  # 4. type
        pages,  # 5. page
        paths,  # 6. path
        contents,  # 7. content
        sources,  # 8. source
        parent_ids  # 9. parent_id
    ]
    # collection.insert(contents)

    # 500개씩 쪼개서(Batch) 안전하게 적재하도록 수정합니다.
    batch_size = 500
    total_count = len(chunk_list)
    print(f"📦 총 {total_count}개의 통합 데이터를 {batch_size}개씩 나누어 Milvus에 적재합니다...")

    for i in range(0, total_count, batch_size):
        # contents 안의 각 리스트(id, text, 벡터 등)를 batch_size만큼 자르기
        batch = [col[i: i + batch_size] for col in milvus_batch_data]
        collection.insert(batch)
        print(f"  -> {min(i + batch_size, total_count)} / {total_count} 적재 완료...")
    print(f"--- [Milvus] {total_count}개 데이터 분할 적재 완료! 🚀 ---")

    #Dense 벡터 인덱스 (COSINE 유사도 사용)
    dense_index = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 200}}
    collection.create_index("dense_vector", dense_index)  # 👈 여기 수정

    # Sparse 벡터 인덱스 (IP - 내적 유사도 사용)
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {"drop_ratio_build": 0.2}}
    collection.create_index("sparse_vector", sparse_index)

    collection.load()

    print(f"--- [Step 7] 밀버스 {len(chunk_list)}개 데이터 적재 완료: {time.time() - start_time:.2f}초 소요! 🚀 ---")
    return collection


# # Fixed-length Chunking 사용
# def milvus_custom_fl(chunk_list, contents, vectors, collection_name, source_name):
#
#     # 1. 서버 연결 및 DB 생성
#     load_dotenv()
#     MILVUS_HOST = os.environ.get("MILVUS_HOST")  # "127.0.0.1"
#     MILVUS_PORT = os.environ.get("MILVUS_PORT")  # "19530"
#     connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
#
#     if "edu" not in db.list_database():
#         db.create_database("edu")
#     db.using_database("edu")
#
#     # 2. 스키마 설계
#     fields = [
#         # PK: 숫자형 자동 생성
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         # 벡터 필드: BGE-M3 (1024 dim)
#         FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
#         # 메타데이터 필드
#         FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
#         FieldSchema(name="chunk_no", dtype=DataType.INT64),
#         FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
#         FieldSchema(name="context_type", dtype=DataType.VARCHAR, max_length=50),
#         FieldSchema(name="page_numbers", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)
#     ]
#
#     if utility.has_collection(collection_name):
#         utility.drop_collection(collection_name)
#     schema = CollectionSchema(fields=fields, description="순정 방식 적재")
#     collection = Collection(name=collection_name, schema=schema)
#
#     # 3. 메타데이터 조립 (contents와 vectors는 앞에서 이미 만들어옴!)
#     sources = [source_name] * len(chunk_list)
#     metas = [chunk for chunk in chunk_list]
#
#     # 4. 삽입 및 인덱스 생성
#     collection.insert([contents, vectors, sources, metas])
#     collection.create_index(field_name="vector", index_params={"index_type": "FLAT", "metric_type": "COSINE"})
#     collection.load()
#
#     print(f"--- [Milvus] {len(chunk_list)}개 데이터 적재 완료! 🚀 ---")
#     return collection
#
# # Parent-Child Chunking 사용
# def milvus_custom_pc(chunk_list, contents, vectors, collection_name, source_name):
#
#     # 1. 서버 연결 및 DB 생성
#     load_dotenv()
#     MILVUS_HOST = os.environ.get("MILVUS_HOST")  # "127.0.0.1"
#     MILVUS_PORT = os.environ.get("MILVUS_PORT")  # "19530"
#     connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
#     if "edu" not in db.list_database():
#         db.create_database("edu")
#     db.using_database("edu")
#
#     # 2. 스키마 설계
#     fields = [
#         # PK: 고유 식별자
#         FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         # 검색 대상 텍스트 (Child)
#         FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8000),
#         # BGE-M3 임베딩 벡터
#         FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
#         # 파일 소스 (필터링 최적화를 위해 partition_key 설정 추천)
#         FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500, partition_key=True),
#         # 가변 메타데이터 (parent_id, page_numbers, context_type 등)
#         FieldSchema(name="dl_meta", dtype=DataType.JSON)
#     ]
#
#     if utility.has_collection(collection_name):
#         utility.drop_collection(collection_name)
#     schema = CollectionSchema(fields=fields, description="순정 방식 적재")
#     collection = Collection(name=collection_name, schema=schema)
#
#     # 3. 메타데이터 조립 (contents와 vectors는 앞에서 이미 만들어옴!)
#     sources = [source_name] * len(chunk_list)
#     metas = [chunk for chunk in chunk_list]
#
#     # 4. 삽입 및 인덱스 생성
#     collection.insert([contents, vectors, sources, metas])
#     collection.create_index(field_name="vector", index_params={"index_type": "FLAT", "metric_type": "COSINE"})
#     collection.load()
#
#     print(f"--- [Milvus] {len(chunk_list)}개 데이터 적재 완료! 🚀 ---")
#     return collection