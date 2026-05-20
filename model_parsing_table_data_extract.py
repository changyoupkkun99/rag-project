import time, json
import pandas as pd
from pathlib import Path

# 표 추출 및 내보내기 함수 정의
def extract_and_tables(input_conv, output_directory, ready_path):
    """
    PDF 문서에서 표를 추출하고 CSV 및 HTML 형식으로 내보내는 함수

    Args:
        input_path (Path): 입력 PDF 파일 경로
        output_directory (Path): 출력 파일을 저장할 디렉토리 경로

    Returns:
        int: 추출된 표의 수
    """
    # 0. 환경 설정
    # 파일명 추출
    doc_filename = input_conv.input.file.stem
    # 출력 디렉토리가 없으면 생성
    table_save_path = output_directory / "tables"
    table_save_path.mkdir(parents=True, exist_ok=True)
    # 변환 시작 시간 기록
    start_time = time.time()
    saved_table_metadata = []  # 나중에 Milvus 연동을 위한 메타데이터 리스트
    start_time2 = time.time()
    # 표 내보내기
    for i, table in enumerate(input_conv.document.tables): # input_conv.document.tables에 이미 표 데이터가 정리되어 있음
        table_df = table.export_to_dataframe(doc=input_conv.document) # Pandas DataFrame으로 변환 (메모리상 데이터)
        # korsquad는 대량의 데이터라 주석처리 해둠
        """
        # 표를 CSV로 저장
        csv_filename = table_save_path / f"{doc_filename}-table-{i + 1}.csv"
        table_df.to_csv(csv_filename, index=False)

        # 표를 HTML로 저장
        html_filename = table_save_path / f"{doc_filename}-table-{i + 1}.html"
        with html_filename.open("w", encoding="utf-8") as fp:
            fp.write(table.export_to_html(doc=input_conv.document))
        """
        # 표를 '| 항목 | 값 |' 형태의 텍스트로 변환합니다.
        markdown_text = table_df.to_markdown(index=False)

        # 메타데이터 수집
        saved_table_metadata.append({
            "id": table.self_ref,  # Docling 고유 ID
            "type": "table_text",  # 타입: 표 텍스트(마크다운)
            "page": table.prov[0].page_no if table.prov else None,
            "path": None,  # 물리적 파일이 없으므로 None (또는 "")
            # "path": str(csv_filename),  # csv 필요 시 주석 해제
            # "path": str(html_filename), # HTML 필요 시 주석 해제
            "content": f"[Table {i + 1} on Page {table.prov[0].page_no if table.prov else '?'}]\n{markdown_text}",
            "source": doc_filename,  # 원본 문서명
            "parent_id": None  # 부모 계층 (표 자체이므로 None)
        })
        if (i + 1) % 100 == 0:  # 출력이 너무 많으니 100단위로 변경
            print(f"  👉 채점 진행률: {i + 1}/{len(input_conv.document.tables)}({time.time() - start_time2:.2f}초 소요")
            start_time2 = time.time()
    ready_json_path = ready_path / "parsing_3_ready_data.json"
    with open(ready_json_path, "w", encoding="utf-8") as f:
        json.dump(saved_table_metadata, f, ensure_ascii=False, indent=4)

    # 변환 소요 시간 계산
    elapsed_time = time.time() - start_time
    print(f"--- [Step 3] 표 데이터 추출 완료! ({elapsed_time:.2f}초 소요) ---")
    # 추출된 표 수 확인
    print(f"총 추출된 표 데이터: {len(saved_table_metadata)}세트")

    return saved_table_metadata