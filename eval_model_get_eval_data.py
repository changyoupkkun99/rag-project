import os, time, json, torch, gc
from dotenv import load_dotenv
from datasets import load_dataset
import model_retrieval  # 💡 전역변수 해제를 위해 임포트
from model_retrieval import execute_hybrid_retrieval


def load_and_retrieve_eval_data(collection_name, ready_path):
    """
    HuggingFace datasets를 이용해 질의-응답 쌍을 추출하고,
    Milvus에서 검색한 결과까지 합쳐서 JSON으로 영구 저장합니다.
    """
    start_time = time.time()
    load_dotenv()
    SEED = int(os.environ.get("SEED", 42))
    SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 1000))  # 2.0은 너무 크니 1000개로 제한

    if "1_0" in collection_name or "1.0" in collection_name:
        dataset = load_dataset("squad_kor_v1", split="validation")
    elif "2_0" in collection_name or "2.0" in collection_name:
        dataset = load_dataset("LGCNS/KorQuAD_2.0", split="validation")
        dataset = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
    else:
        raise ValueError(f"❌ 지원하지 않는 컬렉션 이름입니다: {collection_name}")

    query_only_data = []  # 질문 + 정답만 (순수 데이터)
    retrieval_ready_data = []  # 질문 + 정답 + 검색결과 (평가용 종합 데이터)

    print(f"✅ 총 {len(dataset)}개의 질의-응답 쌍 로드 완료! 🔍 Milvus 검색을 시작합니다.(소요 시간: {time.time() - start_time:.2f}초)")
    start_time_2 = time.time()
    # 💡 데이터 평탄화 및 검색 수행 루프
    for idx, row in enumerate(dataset):
        question = row['question']

        # 정답 추출 (1.0 리스트 vs 2.0 단일 정답 대응)
        answers_list = []
        if 'answers' in row:
            answers_list = row['answers']['text']  # 여러 개의 정답 리스트 그대로 유지
        elif 'answer' in row:
            ans = row['answer']['text'] if isinstance(row['answer'], dict) else row['answer']
            answers_list = [ans]

        # 1. 순수 질문-정답 데이터 구성
        query_info = {
            "id": f"Q_{idx + 1}",
            "question": question,
            "answers": answers_list
        }
        query_only_data.append(query_info)

        # 2. Milvus 하이브리드 검색 수행 (이전 함수 호출)
        retrieved_docs = execute_hybrid_retrieval(question, collection_name)

        # 3. 검색 결과를 포함한 종합 데이터 구성
        retrieval_info = {
            "id": f"Q_{idx + 1}",
            "question": question,
            "answers": answers_list,
            "retrieved_docs": retrieved_docs
        }
        retrieval_ready_data.append(retrieval_info)

        # 🌟 tqdm 대신 10단위 깔끔한 진행률 출력
        if (idx + 1) % 10 == 0:
            print(f"  👉 검색 진행률: {idx + 1}/{len(dataset)} (⏱️ 소요 시간: {time.time() - start_time_2:.2f}초)")
            start_time_2 = time.time()

    # ---------------------------------------------------------
    # 4. JSON 파일로 두 가지 버전 모두 저장
    # ---------------------------------------------------------

    query_json_path = ready_path / "query_ready_data.json"
    retrieval_json_path = ready_path / "retrieval_ready_data.json"

    with open(query_json_path, "w", encoding="utf-8") as f_json:
        json.dump(query_only_data, f_json, ensure_ascii=False, indent=4)

    with open(retrieval_json_path, "w", encoding="utf-8") as f_json:
        json.dump(retrieval_ready_data, f_json, ensure_ascii=False, indent=4)

    print(f"\n💾 JSON 캐싱 완료! 이제 검색 없이 이 파일들로 평가를 진행할 수 있습니다.(⏱️ 총 소요 시간: {time.time() - start_time:.2f}초)")
    print(f"  - 쿼리 원본: {query_json_path}")
    print(f"  - 검색 포함본: {retrieval_json_path}")

    # vram 청소
    print("\n🧹 검색 종료! VRAM 청소.")
    if hasattr(model_retrieval, 'embed_model'):
        model_retrieval.embed_model = None
    if hasattr(model_retrieval, 'rerank_model'):
        model_retrieval.rerank_model = None

    gc.collect()
    torch.cuda.empty_cache()
    print("✅ VRAM 청소 완료! 이제 GPU가 깨끗해졌습니다.")
    return query_only_data, retrieval_ready_data