import os, time, json, gc, torch
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate

prompt = None
llm = None


def generate_rag_answer(query, retrieved_docs):
    """
    검색된 문서(retrieved_docs)를 바탕으로 LlamaIndex LLM을 통해 답변을 생성합니다.
    """
    global prompt, llm

    # 💡 [Lazy Loading] 최초 1회 실행 시에만 모델 로드
    if llm is None:
        print("\n⚡ [Lazy Load] Ollama LLM 클라이언트 연결 및 프롬프트 로드 중...")
        load_dotenv()
        RAG_PROMPT_TEMPLATE = os.environ.get("RAG_PROMPT_TEMPLATE")
        prompt = PromptTemplate(RAG_PROMPT_TEMPLATE)

        # LlamaIndex용 Ollama 세팅 (원하시는 모델명으로 수정 가능)
        llm = Ollama(model="gemma4:e4b", temperature=0.0, request_timeout=600.0)
        # llm = Ollama(model="gemma3:4b", temperature=0.0, request_timeout=600.0) # Gemma3

    # 1. 문서 포맷팅 (리스트 -> 단일 텍스트 문자열)
    formatted_text = ""
    for doc in retrieved_docs:
        # 검색 단계에서 수정한 스키마에 맞게 안전하게 추출
        metadata = doc.get('metadata', {})
        source_id = metadata.get('parent_id') or metadata.get('id') or 'Unknown'
        formatted_text += f"[Source: {source_id}]\n{doc['page_content']}\n\n"

    # 2. 프롬프트 완성
    fmt_prompt = prompt.format(context=formatted_text, input=query)

    # 3. 답변 생성
    response = llm.complete(fmt_prompt)

    return response.text


def llm_answer_generation(retrieved_data, ready_path):
    """
    검색된 데이터 전체를 돌며 LLM 답변을 생성하고 JSON으로 영구 저장합니다.
    """
    start_time = time.time()
    print(f"\n🚀 LLM 답변 생성 시작! (총 {len(retrieved_data)}개 쿼리)")
    generation_ready_data = []

    for idx, item in enumerate(retrieved_data):
        query = item["question"]

        # 토큰 절약을 위해 상위 3~5개만 넣는 것을 권장합니다 (선택사항)
        # docs_for_llm = item["retrieved_docs"][:3]
        docs_for_llm = item["retrieved_docs"]

        # LLM 호출
        generated_answer = generate_rag_answer(query, docs_for_llm)

        # 기존 데이터에 'generated_answer' 키만 새로 추가해서 복사
        gen_item = item.copy()
        gen_item["generated_answer"] = generated_answer
        generation_ready_data.append(gen_item)

        if (idx + 1) % 10 == 0:
            print(f"  👉 답변 생성 진행률: {idx + 1}/{len(retrieved_data)} (⏱️ 소요 시간: {time.time() - start_time:.2f}초)")
            # 10번마다 VRAM 청소
            torch.cuda.empty_cache()
            gc.collect()
            start_time = time.time()  # 다음 10개를 위해 시간 초기화

    # 4. 완료 후 JSON 저장
    generation_json_path = ready_path / "generation_ready_data.json"
    with open(generation_json_path, "w", encoding="utf-8") as f:
        json.dump(generation_ready_data, f, ensure_ascii=False, indent=4)

    print(f"\n💾 답변 생성 및 캐싱 완료! -> {generation_json_path}")

    # 5. 최종 VRAM 청소
    global llm
    llm = None
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ LLM VRAM 청소 완료! 이제 편하게 JSON을 열어 확인해보세요.")

    return generation_ready_data