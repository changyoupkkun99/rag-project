import time, json
from docling_core.types.doc import PictureItem, TableItem

def extract_and_images(input_conv, output_directory, ready_path):
    # 환경 설정
    """
    PDF 문서에서 이미지(Images)를 추출하고 PNG 형식으로 저장하는 함수
    """
    # 파일명 추출
    doc_filename = input_conv.input.file.stem
    # 출력 디렉토리가 없으면 생성
    image_save_path = output_directory / "images"
    image_save_path.mkdir(parents=True, exist_ok=True)
    # 변환 시작 시간 기록
    start_time = time.time()
    saved_image_paths = []  # 저장된 경로를 담을 리스트

    """
    # 이미지 페이지 저장
    for page_no, page in input_conv.document.pages.items():
        page_no = page.page_no
        page_image_filename = image_save_path / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")
    """

    # 이미지 및 테이블 저장
    table_counter = 0
    picture_counter = 0
    for element, _level in input_conv.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            img = element.get_image(input_conv.document) # 표 이미지 유/무 확인
            if img:
                element_image_filename = (
                        image_save_path / f"{doc_filename}-table-{table_counter}.png"
                )
                with element_image_filename.open("wb") as fp:
                    element.get_image(input_conv.document).save(fp, "PNG")
                # caption_text 표는 사진과 다르게 text로 넣어지기 때문에 필요없지만 형식 통일을 위해 넣음
                page_num = element.prov[0].page_no if element.prov else "?"
                caption_text = f"[Table {table_counter}] 이 이미지는 {page_num}페이지에 포함된 데이터 표입니다. (파일명: {element_image_filename.name})"

                saved_image_paths.append({
                    "id": element.self_ref,  # Docling 고유 ID
                    "type": "table_image",  # 타입: 표 이미지
                    "page": element.prov[0].page_no if element.prov else None,
                    "path": str(element_image_filename),  # 저장된 이미지 경로
                    "content": caption_text,  # 캡션 (예: "[Table 1] 이 이미지는...")
                    "source": doc_filename,  # 원본 문서명
                    "parent_id": None  # 부모 계층 (이미지는 없으므로 None)
                })
        if isinstance(element, PictureItem):
            picture_counter += 1
            img = element.get_image(input_conv.document)  # 사진 이미지 유/무 확인
            if img:
                element_image_filename = (
                        image_save_path / f"{doc_filename}-picture-{picture_counter}.png"
                )
                with element_image_filename.open("wb") as fp:
                    element.get_image(input_conv.document).save(fp, "PNG")

                # 사진이 검색될 수 있도록 텍스트 설명(Caption) 추가
                page_num = element.prov[0].page_no if element.prov else "?"
                caption_text = f"[Figure {picture_counter}] 이 이미지는 {page_num}페이지에 포함된 시각 자료입니다. (파일명: {element_image_filename.name})"

                saved_image_paths.append({
                    "id": element.self_ref,  # Docling 고유 ID
                    "type": "picture",  # 타입: 일반 사진
                    "page": element.prov[0].page_no if element.prov else None,
                    "path": str(element_image_filename),  # 저장된 이미지 경로
                    "content": caption_text,  # 캡션 (예: "[Figure 1] 이 이미지는...")
                    "source": doc_filename,  # 원본 문서명
                    "parent_id": None  # 부모 계층 (사진은 없으므로 None)
                })
    """ 자동화
    # 이미지가 내장된(Base64) 마크다운으로 저장 (파일 하나에 이미지 데이터 포함)
    md_filename = output_directory / f"{doc_filename}-with-images.md"
    input_conv.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # 외부 참조 이미지를 포함한 마크다운으로 저장 (MD 파일 + images 폴더)
    md_filename = output_directory / f"{doc_filename}-with-image-refs.md"
    input_conv.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # 외부 참조 이미지를 포함한 HTML로 저장 (HTML 파일 + images 폴더)
    html_filename = output_directory / f"{doc_filename}-with-image-refs.html"
    input_conv.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
    """
    ready_json_path = ready_path / "parsing_2_ready_data.json"
    with open(ready_json_path, "w", encoding="utf-8") as f:
        json.dump(saved_image_paths, f, ensure_ascii=False, indent=4)

    # 변환 소요 시간 계산
    elapsed_time = time.time() - start_time
    print(f"--- [Step 2] Figure/Table 이미지 추출 완료!({elapsed_time:.2f}초 소요) ---")

    # 총 추출된 이미지(Figure+Table) 수 확인
    total_image = len(input_conv.document.pictures) + len(input_conv.document.tables)
    print(f"총 추출된 PNG 파일 수: {total_image}개 (표: {table_counter}, 그림: {picture_counter})")
    return saved_image_paths


# # --- 실행부 ---
# # 사용자 경로 설정 (리눅스 환경일 경우 경로 형식을 /home/lcy/... 로 수정하세요)
# source = "./RAG_자료조사_25.11.21.pdf" # 파일 경로
# pdf_name = Path(source).stem # pdf 이름 추출
# output_path = Path(f"output/{pdf_name}")
# output_path.mkdir(parents=True, exist_ok=True)
#
# try:
#     count = extract_and_images(source, output_path)
#     print(f"성공: {count}개의 이미지를 {output_path}에 저장했습니다.")
# except Exception as e:
#     print(f"이미지 추출 중 오류 발생: {e}")