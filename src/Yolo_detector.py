import pymupdf as fitz
from PIL import Image
from pathlib import Path
import numpy as np
import io
import torch
import json
import logging
from transformers import NougatProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
from typing import List, Dict, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PDF_RENDER_DPI = 200

NOUGAT_MODEL_NAME = "facebook/nougat-base" 
LAYOUT_MODEL_PATH = "./yolov8l-doclaynet.pt"


def group_and_sort_blocks_by_columns(page_blocks: List[Dict], page_width: int) -> List[Dict]:
   
    if not page_blocks:
        return []

    page_center = page_width / 2
    full_width_threshold = page_width * 0.6 

    full_width_blocks = []
    left_column = []
    right_column = []

    for block in page_blocks:
        bbox = block.get("bbox")
        if not bbox: continue
        
        block_width = bbox[2] - bbox[0]
        block_center_x = (bbox[0] + bbox[2]) / 2

        if block_width > full_width_threshold:
            full_width_blocks.append(block)
        elif block_center_x < page_center:
            left_column.append(block)
        else:
            right_column.append(block)
    
   
    full_width_blocks.sort(key=lambda b: b['bbox'][1])
    left_column.sort(key=lambda b: b['bbox'][1])
    right_column.sort(key=lambda b: b['bbox'][1])
    
    return full_width_blocks + left_column + right_column

class YoloPDFContentExtractor:

    def __init__(self):
        logging.info(f"Используется устройство: {DEVICE}")
        if DEVICE == "cpu":
            logging.warning("ВНИМАНИЕ: Выполнение на CPU будет медленным.")
        self.load_models()

    def load_models(self):
        logging.info("Загрузка моделей...")
        try:
            self.layout_model = YOLO(LAYOUT_MODEL_PATH)
            self.layout_model.to(DEVICE)
            self.id2label = self.layout_model.names
            logging.info(f"Модель макета YOLO '{LAYOUT_MODEL_PATH}' загружена.")
            logging.info(f"Классы модели: {self.id2label}")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели YOLO из '{LAYOUT_MODEL_PATH}'. Убедитесь, что файл скачан и находится в этой папке. Ошибка: {e}")
            raise
        self.nougat_processor = NougatProcessor.from_pretrained(NOUGAT_MODEL_NAME)
        self.nougat_model = VisionEncoderDecoderModel.from_pretrained(NOUGAT_MODEL_NAME).to(DEVICE)
        logging.info(f"Модель формул '{NOUGAT_MODEL_NAME}' загружена.")
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['ru', 'en'])
            logging.info("Модель текста 'EasyOCR' загружена.")
        except ImportError:
            logging.error("EasyOCR не установлен. Пожалуйста, установите его: pip install easyocr")
            self.ocr_reader = None

    def _recognize_formula(self, image: Image.Image) -> str:
        pixel_values = self.nougat_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            generated_ids = self.nougat_model.generate(
                pixel_values,
                decoder_input_ids=torch.tensor([[self.nougat_processor.tokenizer.bos_token_id]]).to(DEVICE),
                max_length=512
            )
        return self.nougat_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


    def process_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        
        pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
        img_pil = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
        page_blocks = []

        logging.info("Определение макета страницы с помощью YOLO...")
        predictions = self.layout_model(img_pil, conf=0.6) 

        for result in predictions:
            boxes = result.boxes
            for box in boxes:
                x1_block, y1_block, x2_block, y2_block = [int(i) for i in box.xyxy[0]]
                class_id = int(box.cls[0])
                class_name = self.id2label[class_id]
                
                cropped_img = img_pil.crop((x1_block, y1_block, x2_block, y2_block))
                
                if class_name == 'Formula':
                    formula_text = self._recognize_formula(cropped_img)
                    if formula_text:
                        page_blocks.append({
                            "type": "formula", "content": formula_text, "bbox": [x1_block, y1_block, x2_block, y2_block], "page_num": page_num
                        })
                elif class_name in ['Text', 'Title', 'List-item', 'Caption', 'Section-header', 'Footnote']:
                     if self.ocr_reader:
                        image_np = np.array(cropped_img)
                        line_results = self.ocr_reader.readtext(image_np, detail=1, paragraph=False)

                        if line_results:
                            logging.info(f"Блок '{class_name}': найдено {len(line_results)} строк текста.")
                            for (bbox, text, conf) in line_results:
                                (tl, tr, br, bl) = bbox
                                x1_line, y1_line = int(tl[0]), int(tl[1])
                                x2_line, y2_line = int(br[0]), int(br[1])

                                abs_x1 = x1_block + x1_line
                                abs_y1 = y1_block + y1_line
                                abs_x2 = x1_block + x2_line
                                abs_y2 = y1_block + y2_line
                                
                                page_blocks.append({
                                    "type": "text", 
                                    "content": text.strip(), 
                                    "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                                    "page_num": page_num
                                })
                else: 
                    page_blocks.append({
                        "type": class_name.lower(), "content": None, "bbox": [x1_block, y1_block, x2_block, y2_block], "page_num": page_num
                    })
        return page_blocks

    def extract_content_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        all_blocks = []
        for page_num, page in enumerate(doc):
            logging.info(f"\n--- Обработка страницы {page_num + 1}/{len(doc)} ---")
            
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
            page_width = pix.width
            
            page_blocks = self.process_page(page, page_num + 1)
            sorted_page_blocks = group_and_sort_blocks_by_columns(page_blocks, page_width)
            
            all_blocks.extend(sorted_page_blocks)
            
        return all_blocks

    def extract_content_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        all_blocks = []
        for page_num, page in enumerate(doc):
            logging.info(f"\n--- Обработка страницы {page_num + 1}/{len(doc)} ---")
            page_img_width = int(page.rect.width * PDF_RENDER_DPI / 72)
            page_blocks = self.process_page(page, page_num + 1)
            sorted_page_blocks = sorted(page_blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))
            all_blocks.extend(sorted_page_blocks)
        return all_blocks

if __name__ == "__main__":
    INPUT_DIR = Path("./input_pdfs")
    OUTPUT_DIR = Path("./output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_file = INPUT_DIR / "example_article23.pdf"
    
    if not Path(LAYOUT_MODEL_PATH).exists():
         logging.error(f"Модель {LAYOUT_MODEL_PATH} не найдена. Пожалуйста, скачайте ее со страницы hantian/yolo-doclaynet")
    elif not pdf_file.exists():
        logging.error(f"Файл не найден: {pdf_file}.")
    else:
        json_output_file = OUTPUT_DIR / f"{pdf_file.stem}_yolo.json"
        txt_output_file = OUTPUT_DIR / f"{pdf_file.stem}_yolo_tts.txt"
        extractor = YoloPDFContentExtractor()
        blocks = extractor.extract_content_from_pdf(pdf_file)
        with open(json_output_file, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)
        logging.info(f"JSON сохранен: {json_output_file}")