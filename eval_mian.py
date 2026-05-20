import os, json
from pathlib import Path
from dotenv import load_dotenv

# 사용자님의 하이브리드 파이프라인 모듈
from eval_model_get_eval_data import load_and_retrieve_eval_data
from eval_model_mrr import evaluate_mrr_from_json
from eval_model_ragas_kor import llm_answer_generation

def main_eval():
    load_dotenv()
    collection_name = os.environ.get("CURRENT_COLLECTION_NAME")
    if not collection_name:
        print("❌ 환경 변수에 컬렉션(CURRENT_COLLECTION_NAME) 이름이 없습니다.")
        return
    ready_path = Path(f"output/{collection_name}/ready/")
    ready_path.mkdir(parents=True, exist_ok=True)

    query_save_json = ready_path / "query_ready_data.json"
    retrieval_save_json = ready_path / "retrieval_ready_data.json"
    generation_save_json = ready_path / "generation_ready_data.json"

    # 평가 데이터 가져오기
    if query_save_json.exists() and retrieval_save_json.exists():
        print(f"이미 {query_save_json} and {retrieval_save_json}  존재")
        with open(query_save_json, "r", encoding="utf-8") as f:
            query_data = json.load(f)
        with open(retrieval_save_json, "r", encoding="utf-8") as f:
            retrieved_data = json.load(f)
    else:
        query_data, retrieved_data = load_and_retrieve_eval_data(collection_name, ready_path)

    # MRR 평가
    evaluate_mrr_from_json(retrieved_data, collection_name)
    # RAGAS 평가
    if generation_save_json.exists():
        print(f"이미 {generation_save_json} 존재")
        with open(generation_save_json, "r", encoding="utf-8") as f:
            generation_data = json.load(f)
    else:
        generation_data = llm_answer_generation(retrieved_data, ready_path)


    # evaluate_ragas(ragas_data_dict)




if __name__ == "__main__":
    main_eval()