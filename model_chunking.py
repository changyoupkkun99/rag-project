import os, json, time
from docling.chunking import HybridChunker
from dotenv import load_dotenv

# Fixed-length Chunking
def fixed_length_chunking_docling(input_conv, output_directory, ready_path):
    """
    [Step 1-1] 파싱된 문서 객체를 BGE-M3 토크나이저 기준으로 자르는 함수
    """
    # 환경 설정
    start_time = time.time()
    load_dotenv()
    EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID")  # 임베딩 모델
    chunk_size = int(os.environ.get("FL_CHUNK_SIZE", 256))

    doc_filename = input_conv.input.file.stem
    chunking_save_path_md = output_directory / f"{doc_filename}_chunks_fl.md"

    # 하이브리드 청커 설정 (BGE-M3 자를 들고 구조를 보며 자름)
    chunker = HybridChunker(
        tokenizer=EMBED_MODEL_ID, # 필요에 따라 토크나이저 설정, 임베딩 모델, BAAI/bge-m3 -> 한글을 잘 확인
        max_tokens=chunk_size, # 청크 크기 설정
    )

    # 청크 생성 (input_conv.document를 바로 사용)
    chunk_iter = chunker.chunk(input_conv.document)
    # 청크를 리스트로 변환
    chunks = list(chunk_iter)

    # 저장된 경로를 담을 리스트
    chunk_metadata_list = []
    # MD 파일 작성 및 데이터 수집
    with open(chunking_save_path_md, "w", encoding="utf-8") as f_md:
        f_md.write(f"# Fixed-length Chunking Result: {doc_filename}\n")
        f_md.write(f"- Total chunks: {len(chunks)}\n")
        f_md.write(f"- Config: Size {chunk_size}\n")
        f_md.write("=" * 88 + "\n\n")

        for i, chunk in enumerate(chunks):
            # 페이지 번호 추출
            page_num = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items and chunk.meta.doc_items[0].prov else None
            # 리스트에 담기 (나중에 리턴해서 Milvus에 쓸 용도)
            chunk_data = {
                "id": f"FL_{i + 1}",  # 고유 ID 생성
                "type": "chunk_fl",  # 타입: 고정 길이 텍스트
                "page": page_num,
                "path": None,  # 텍스트이므로 파일 경로 없음
                "content": chunk.text,  # 실제 청크 텍스트
                "source": doc_filename,  # ➕ 원본 문서명 (메타데이터 평탄화)
                "parent_id": None  # ➕ 고정 길이는 맥락 구조가 없으므로 None
            }
            chunk_metadata_list.append(chunk_data)

            # MD 저장용 텍스트 (마크다운 문법 적용)
            f_md.write(f"[Chunk No. {i + 1}] (Pages: {page_num})\n")
            f_md.write("Type: chunk_fl\n")
            f_md.write(f"{chunk.text}\n")
            f_md.write("=" * 88 + "\n\n")

    # JSON 파일 저장 (기계가 읽기 좋은 용도)
    ready_json_path = ready_path / "chunking_fl_ready_data.json"
    with open(ready_json_path, "w", encoding="utf-8") as f_json:
        json.dump(chunk_metadata_list, f_json, ensure_ascii=False, indent=4)

    print(f"--- [Step 4-1] Fixed-length 청킹 완료! (총 {len(chunks)}개 조각, {time.time() - start_time:.2f}초 소요) ---")
    return chunk_metadata_list


# Parent-Child Chunking
def parent_child_chunking_docling(input_conv, output_directory, ready_path):
    """
    [Step 1-2] Parent-Child 구조로 청킹하는 함수
    """
    start_time = time.time()
    load_dotenv()
    EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID")  # 임베딩 모델
    chunk_size = int(os.environ.get("PC_CHUNK_SIZE", 800))

    doc_filename = input_conv.input.file.stem
    chunking_save_path_md = output_directory / f"{doc_filename}_chunks_pc.md"

    # 하이브리드 청커 설정
    chunker = HybridChunker(
        tokenizer=EMBED_MODEL_ID, # 필요에 따라 토크나이저 설정, 임베딩 모델, BAAI/bge-m3 -> 한글을 잘 확인
        max_tokens=chunk_size, # 청크 크기 설정
    )

    # 청크 생성 (input_conv.document를 바로 사용)
    chunk_iter = chunker.chunk(input_conv.document)
    # 청크를 리스트로 변환
    chunks = list(chunk_iter)

    # 저장된 경로를 담을 리스트
    chunk_metadata_list = []
    # MD 파일 작성 및 데이터 수집
    with open(chunking_save_path_md, "w", encoding="utf-8") as f_md:
        f_md.write(f"# Parent-Child Chunking Result: {doc_filename}\n")
        f_md.write(f"- Total chunks: {len(chunks)}\n")
        f_md.write(f"- Config: Size {chunk_size}\n")
        f_md.write("=" * 88 + "\n\n")

        for i, chunk in enumerate(chunks):
            # 페이지 번호 추출
            page_num = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items and chunk.meta.doc_items[0].prov else None
            # [Parent 정보 추출]
            # Docling은 해당 조각이 속한 원본 문단/섹션 정보를 앎
            # 여기서는 해당 조각이 속한 상위 섹션이나 원본 텍스트 전체를 Parent로 정의
            # Docling이 파악한 해당 청크의 상위 제목(Heading)을 가져옵니다.
            headings = chunk.meta.headings
            if headings:
                # 상위 제목이 있다면 그것을 부모 ID(혹은 이름)로 사용
                parent_id = f"Section: {headings[0]}"
            else:
                # 상위 제목이 없는 파편화된 조각이면 문서 이름 자체를 부모로 설정
                parent_id = f"Doc: {doc_filename}"

            # 데이터 수집
            chunk_data = {
                "id": f"PC_{i + 1}", # Docling ID 또는 직접 생성한 ID
                "type": "chunk_pc",  # 타입: 부모-자식 텍스트
                "page": page_num, # 페이지 번호
                "path": None, # 물리적 파일이 없으므로 None
                "content": chunk.text,  # 실제 청크 텍스트
                "source": doc_filename,  # 원본 문서명
                "parent_id": parent_id  # 부모 노드 정보
            }
            chunk_metadata_list.append(chunk_data)

            # MD 파일 작성 (시각적 구분)
            f_md.write(f"[Chunk No. {i + 1}] (Page: {page_num} | Parent: {parent_id})\n")
            f_md.write(f"Type: chunk_pc\n")
            f_md.write(f"{chunk.text}\n")
            f_md.write("=" * 88 + "\n\n")

    # JSON 저장
    ready_json_path = ready_path / "chunking_pc_ready_data.json"
    with open(ready_json_path, "w", encoding="utf-8") as f_json:
        json.dump(chunk_metadata_list, f_json, ensure_ascii=False, indent=4)

    print(f"--- [Step 4-2] Parent-Child 청킹 완료! (총 {len(chunks)}개 조각, ({time.time() - start_time:.2f}초 소요) ---")
    return chunk_metadata_list