import os, time, json  # 시스템 환경 변수 제어를 위한 모듈
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, HTMLFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline


def parsing_docling(input_path, output_directory, pipeline_options_total):
    start_time = time.time()
    # 0. 환경 설정
    os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/tessdata/"  # conver 오류, Tesseract 언어 데이터(사전)가 있는 물리적 경로 지정
    file_name = Path(input_path).stem # 파일 이름 따오기
    parsing_save_path = output_directory / f"{file_name}_parsing.md" # 신규 파일 이름 및 경로 설정

    # 1. 파싱 설정
    pipeline_options = pipeline_options_total # PdfPipelineOptions 설정 가져오기

    # 2. 컨버터 설정
    converter = DocumentConverter( # Docling을 사용해 변환
        format_options={
            InputFormat.PDF: PdfFormatOption( # pdf 파이프라인 설정
                pipeline_options=pipeline_options # PdfPipelineOptions 설정 입력
                # backend=PyPdfium2DocumentBackend # PDF Backend = pypdfium2 설정(기본이 더 좋음)
            ),
            InputFormat.HTML: HTMLFormatOption( # HTML 변환
                # pipeline_cls=SimplePipeline
                pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(input_path)  # 변환된 문서를 result에 저장
    markdown_content = result.document.export_to_markdown()
    with open(parsing_save_path, "w", encoding="utf-8") as f:  # 출력 값 저장
        f.write(markdown_content)
    print(f"--- [Step 1] 파싱 완료!({time.time() - start_time:.2f}초 소요)---")
    return result



# # --- 실행부 ---
# # 사용자 경로 설정 (리눅스 환경일 경우 경로 형식을 /home/lcy/... 로 수정하세요)
# source = "./RAG_자료조사_25.11.21.pdf" # 파일 경로
# pdf_name = Path(source).stem # pdf 이름 추출
# output_path = Path(f"output/{pdf_name}")
# output_path.mkdir(parents=True, exist_ok=True)
#
# try:
#     parsing_result = parsing_docling(source, output_path)
#     extract_and_images(parsing_result, output_path)
# except Exception as e:
#     print(f"파싱 중 오류 발생: {e}")

