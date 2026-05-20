def evaluate_mrr_from_json(retrieval_ready_data, collection_name):
    """
    이미 검색이 완료된 데이터를 바탕으로 즉시 MRR을 계산합니다.
    """
    rr_scores = []

    for item in retrieval_ready_data:
        ground_truths = item["answers"]  # 리스트 형태의 정답들
        docs = item["retrieved_docs"]

        rank = 0
        for i, doc in enumerate(docs):
            content = doc["page_content"]
            # 여러 정답(1.0) 중 하나라도 검색된 본문에 포함되어 있으면 정답 인정!
            if any(gt in content for gt in ground_truths):
                rank = i + 1
                break

        # 정답을 찾았을 경우에만 역수 점수를 누적
        if rank > 0:
            rr_scores.append(1.0 / rank)
        else:
            rr_scores.append(0.0)

    total_queries = len(retrieval_ready_data)
    mrr_score = sum(rr_scores) / total_queries if total_queries > 0 else 0.0

    print("=" * 50)
    print(f"🏆 [최종 결과] {collection_name} MRR 점수: {mrr_score:.4f}")
    print("=" * 50)

    return mrr_score