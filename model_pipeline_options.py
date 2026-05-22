import os
import time
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, PipelineOptions


def get_pipeline_options_pdf():
    start_time = time.time()
    pipeline_options = PdfPipelineOptions() # PdfPipelineOptions 활성화

    # PDF 전용: OCR 및 비전 모델(TableFormer) 활성화
    pipeline_options.do_ocr = True # 이미지 내 텍스트 인식을 위해 OCR 활성화, Enable OCR
    pipeline_options.do_table_structure = True  # Pipeline type = Standard, Table Mode 활성화
    lang_env = os.environ.get("OCR_LANG", "kor,eng")  # OCR Engine = Tesseract, OCR Language

    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE # Table Mode = Accurate
    pipeline_options.table_structure_options.do_cell_matching = True # 글자가 선에 걸쳐서 옆 칸으로 새어 나가거나, 증발하는 사고 방지
    # 이미지 추출을 위한 파이프라인 옵션 설정
    pipeline_options.generate_table_images = True
    pipeline_options.generate_picture_images = True # 그림 추출 활성화
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = 2.0

    print(f"--- [Step 0] PDF 파이프라인 설정 완료! ({time.time() - start_time:.2f}초 소요) ---")
    return pipeline_options


def get_pipeline_options_html():
    start_time = time.time()
    # ✨ HTML 전용: 가벼운 기본 PipelineOptions 사용 (PdfPipelineOptions 아님!)
    pipeline_options = PipelineOptions()

    # [KorQuAD 2.0 맞춤 최적화]
    # HTML은 비전 모델 없이 <table> 태그를 직접 100% 정확하게 읽어냅니다.
    # 위키백과 데이터는 쓸데없는 웹 이미지(아이콘, 링크 썸네일 등)가 너무 많아
    # 다운로드 및 추출을 끄면 파싱 속도가 엄청나게 빨라집니다.
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_page_images = False

    print(f"--- [Step 0] HTML 파이프라인 설정 완료! ({time.time() - start_time:.2f}초 소요) ---")
    return pipeline_options