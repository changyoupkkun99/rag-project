import time, json
def summarize_all_chunks(fl_chunks, pc_chunks, table_data, visual_data, doc_file_name):
    """
    파일명과 구조 정보를 활용해 검색 최적화 헤더를 자동으로 생성하고 통합합니다.
    """
    start_time = time.time()
    table_chunks = []
    for t in table_data:
        # 시맨틱 앵커: LLM이 데이터를 명확히 인지할 수 있도록 헤더 결합
        enhanced_table_text = (
            f"### [STRUCTURED_DATA] Source: {t['source']} | ID: {t['id']}\n"
            f"이 섹션은 {t['source']} 문서의 {t['page']}페이지에 있는 정형 데이터 표입니다.\n"
            f"--- DATA START ---\n"
            f"{t['content']}"  # 기존 마크다운 텍스트
        )

        table_chunks.append({
            "id": t["id"],
            "type": t["type"],  # "table_text"
            "page": t["page"],
            "path": t["path"],  # None
            "content": enhanced_table_text,  # 보강된 텍스트로 덮어쓰기
            "source": t["source"],
            "parent_id": t["parent_id"]
        })

    figure_chunks = []
    for img in visual_data:
        enhanced_figure_text = (
            f"### [VISUAL_DATA] Source: {img['source']} | Page: {img['page']}\n"
            f"설명: {img['content']}\n"  # 이전에 만들어둔 caption 텍스트
            f"이미지 경로: {img['path']}"
        )
        # 7개 Key 스키마 유지
        figure_chunks.append({
            "id": img["id"],
            "type": img["type"],  # "picture" 또는 "table_image"
            "page": img["page"],
            "path": img["path"],
            "content": enhanced_figure_text,  # 보강된 텍스트로 덮어쓰기
            "source": img["source"],
            "parent_id": img["parent_id"]
        })

    # 모든 리스트 통합
    total_list = fl_chunks + pc_chunks + table_chunks + figure_chunks
    print(f"--- [Step 5] 통합 완료!, {time.time() - start_time:.2f}초 소요) ---")
    print(f"  - 일반/부모-자식 청크: {len(fl_chunks) + len(pc_chunks)}개")
    print(f"  - 표(Table) 데이터: {len(table_chunks)}개")
    print(f"  - 시각 자료(Figure/Table Image): {len(figure_chunks)}개")
    print(f"  - 🌟 총합 데이터 개수: {len(total_list)}개")

    return total_list


def summarize_all_chunks_finance(fl_chunks, pc_chunks, table_data, visual_data, pdf_name):
    """
    [금융/정책 도메인 특화 버전]
    일반 텍스트, 표, 그림에 '예산, 보증, 수치' 등 금융 특화 키워드를 강제로 주입하여
    하이브리드 검색(Sparse Vector) 성능을 극대화합니다.
    """
    print(f"\n✨ [Chunk Sum] '{pdf_name}' 금융/정책 도메인 맞춤형 데이터 통합 시작...")

    # 1. 표(Table) 데이터 - 💡 금융 특화 메타데이터 주입
    table_chunks = []
    for t in table_data:
        t_idx = t.get('table_index', '?')

        # Sparse 검색 엔진을 유혹할 '금융/통계 마법의 단어'들 배치
        enhanced_table_text = (
            f"### [금융/정책지표 핵심 데이터] 출처: {pdf_name} | 표 번호: Table_{t_idx}\n"
            f"설명: 이 데이터는 {pdf_name} 문서의 {t['page']}페이지에 수록된 정형 통계 수치 표입니다.\n"
            f"연관 키워드: 금융지원, 보증공급, 예산액, 목표 인원, 실적, 금리, 자금 규모\n"
            f"--- 데이터 시작 ---\n"
            f"{t['table_text']}"
        )

        table_chunks.append({
            "text": enhanced_table_text,
            "dl_meta": {
                "parent_id": f"Table_{t_idx}",
                "type": "table",
                "domain": "finance_policy",  # 메타데이터에도 도메인 명시
                "page": t["page"]
            }
        })

    # 2. 그림(Figure) 데이터 - 💡 금융 사업구조도 특화
    figure_chunks = []
    for img in visual_data:
        if img["type"] == "picture":
            enhanced_figure_text = (
                f"### [금융 시각자료/사업 프로세스 도식화] 출처: {pdf_name} | 페이지: {img['page']}\n"
                f"설명: {img.get('caption', '정책/금융 사업 구조 및 흐름도')}\n"
                f"파일경로: {img['path']}"
            )

            figure_chunks.append({
                "text": enhanced_figure_text,
                "dl_meta": {
                    "parent_id": f"Fig_{img['page']}_{img['id'][:8]}",
                    "type": "figure",
                    "domain": "finance_policy",
                    "path": img["path"],
                    "page": img["page"]
                }
            })

    # 3. 텍스트(Text) 데이터 - 💡 출처 명시
    # 본문 텍스트에도 문서명을 박아주면 질문에 "2026년 업무계획에서~" 라는 말이 들어올 때 점수가 확 뜁니다.
    text_chunks = []
    for c in (fl_chunks + pc_chunks):
        original_text = c.get('text', '')
        # 텍스트 맨 앞에 문서 제목을 꼬리표처럼 달아줍니다.
        c['text'] = f"[문서: {pdf_name}]\n{original_text}"

        # 기존 딕셔너리에 dl_meta가 없으면 생성 후 도메인 추가
        if 'dl_meta' not in c:
            c['dl_meta'] = {}
        c['dl_meta']['domain'] = "finance_policy"
        text_chunks.append(c)

    # 4. 전체 병합
    total_list = text_chunks + table_chunks + figure_chunks
    print(f"✅ 금융 도메인 맞춤형 통합 완료 (총 {len(total_list)}개 청크)")

    return total_list