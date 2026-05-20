import os  # 시스템 환경 변수 제어를 위한 모듈
import time
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

def get_pipeline_options():
    start_time = time.time()
    pipeline_options = PdfPipelineOptions() # PdfPipelineOptions 활성화
    # 파싱 설정
    # OCR options
    pipeline_options.do_table_structure = True # Pipeline type = Standard, Table Mode 활성화
    pipeline_options.do_ocr = True # 이미지 내 텍스트 인식을 위해 OCR 활성화, Enable OCR
    lang_env = os.environ.get("OCR_LANG", "kor,eng") # OCR Engine = Tesseract, OCR Language
    # Table options
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE # Table Mode = Accurate
    pipeline_options.table_structure_options.do_cell_matching = True # 글자가 선에 걸쳐서 옆 칸으로 새어 나가거나, 증발하는 사고 방지
    # 이미지 추출을 위한 파이프라인 옵션 설정
    # pipeline_options.generate_table_images = True
    pipeline_options.generate_picture_images = True  # 그림 추출 활성화
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = 2.0  # 고해상도 설정

    print(f"--- [Step 0] 파이프라인 설정 완료!({time.time() - start_time:.2f}초 소요)---")
    return pipeline_options