import os, re, json, gc
from pathlib import Path
from dotenv import load_dotenv, set_key, find_dotenv
# Parsing
from model_pipeline_options import get_pipeline_options # 파이프 라인 설정
from model_parsing_docling import parsing_docling # Docling, Convert 설정
from model_parsing_figure_extract import extract_and_images # 이미지 + 표 Image 추출 -> model_table_split와 차이점: 이미지로 보여주기 위한 용도
from model_parsing_table_data_extract import extract_and_tables # model_parsing_figure_extract와 차이점: LLM이 데이터를 정확히 읽게 하기 위한 용도
# Chunking
from model_chunking import fixed_length_chunking_docling, parent_child_chunking_docling
from model_chunk_sum import summarize_all_chunks, summarize_all_chunks_finance
# Embedding
from model_embedding import generate_embeddings
from model_milvus import milvus_unified_insert
# Evaluation
from eval_mian import main_eval

source = "korquad_2_0_data.html"
file_name = Path(source).stem # 파일 이름 추출
output_path = Path(f"output/{file_name}")
output_path.mkdir(parents=True, exist_ok=True)

ready_path = Path(f"output/{file_name}/ready/")
ready_path.mkdir(parents=True, exist_ok=True)
# 각 단계 json
parsing_step2_json = ready_path / "parsing_2_ready_data.json"
parsing_step3_json = ready_path / "parsing_3_ready_data.json"
chunking_save_fl_json = ready_path / "chunking_fl_ready_data.json"
chunking_save_pc_json = ready_path / "chunking_pc_ready_data.json"

try:
    # Parsing
    print(f"\n✨ 파이프라인 시작!")
    # [Step 0] 파이프라인 설정 로드
    pipeline_options = get_pipeline_options()
    # [Step 1] 파싱 작업
    conv_result = parsing_docling(source, output_path, pipeline_options)

    if parsing_step2_json.exists():
        print(f"이미 {parsing_step2_json} 존재")
        with open(parsing_step2_json, "r", encoding="utf-8") as f:
            figure_data = json.load(f)
    else:
        figure_data = extract_and_images(conv_result, output_path, ready_path)

    if parsing_step3_json.exists():
        print(f"이미 {parsing_step3_json} 존재")
        with open(parsing_step3_json, "r", encoding="utf-8") as f:
            table_data = json.load(f)
    else:
        table_data = extract_and_tables(conv_result, output_path, ready_path)

    # Chunking
    print(f"\n✨ 청킹 시작!")
    if chunking_save_fl_json.exists():
        print(f"이미 {chunking_save_fl_json} 존재")
        with open(chunking_save_fl_json, "r", encoding="utf-8") as f:
            fl_chunk_data = json.load(f)
    else:
        fl_chunk_data = fixed_length_chunking_docling(conv_result, output_path, ready_path) # Fixed-length Chunking

    if chunking_save_pc_json.exists():
        print(f"이미 {chunking_save_pc_json} 존재")
        with open(chunking_save_pc_json, "r", encoding="utf-8") as f:
            pc_chunk_data = json.load(f)
    else:
        pc_chunk_data = parent_child_chunking_docling(conv_result, output_path, ready_path) # Parent-Child Chunking

    multi_chunk_data= summarize_all_chunks(fl_chunks=fl_chunk_data, pc_chunks=pc_chunk_data, table_data=table_data,visual_data=figure_data, doc_file_name=file_name)

    # Embedding
    print(f"\n✨ 임베딩 및 Milvus 적재 시작!")
    dense_vectors, sparse_vectors = generate_embeddings(multi_chunk_data)
    safe_collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', file_name) + "_db"  # 파일명에서 한글, 특수문자, 마침표를 모두 '_'로 변환
    collection = milvus_unified_insert(
        chunk_list=multi_chunk_data,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        collection_name=safe_collection_name,
        doc_filename=file_name
    )

    # env 자동화 및 기록
    dotenv_path = find_dotenv()  # .env 파일 위치 자동 추적
    load_dotenv(dotenv_path)  # 기존에 있던 값을 일단 읽어옵니다.
    set_key(dotenv_path, "CURRENT_COLLECTION_NAME", safe_collection_name)
    # [Step 1] 히스토리에 누적 저장
    existing_history = os.environ.get("COLLECTION_HISTORY", "")
    # [Step 2] 이미 히스토리에 있는 이름이 아니라면 추가 (중복 방지)
    if safe_collection_name not in existing_history:
        if existing_history:
            new_history = f"{existing_history},{safe_collection_name}"
        else:
            new_history = safe_collection_name

        set_key(dotenv_path, "COLLECTION_HISTORY", new_history)
    # 결과 확인
    print(f"\n✨ env 자동화 및 기록 종료!")

    main_eval()
except Exception as e:
    print(f"❌ 오류: {e}")

