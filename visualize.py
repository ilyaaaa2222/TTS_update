import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict


def _split_text_block_into_lines(image: Image.Image) -> List[Dict]:

    img_cv = np.array(image.convert("L"))
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 15 and h > 8:
            lines.append({"bbox": [x, y, x + w, y + h]})
    lines.sort(key=lambda item: item['bbox'][1])
    return lines


def visualize_detections(pdf_path: Path, page_number: int, yolo_model_path: str, output_path: Path):
    
    logging.info(f"Загрузка модели YOLO из {yolo_model_path}...")
    model = YOLO(yolo_model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Обработка страницы {page_number} из файла {pdf_path}...")
    doc = fitz.open(pdf_path)
    if page_number > len(doc):
        logging.error(f"В документе всего {len(doc)} страниц. Невозможно обработать страницу {page_number}.")
        return

    page = doc[page_number - 1] 
    
    pix = page.get_pixmap(dpi=300)
    img_pil = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
    
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    logging.info("Детекция блоков с помощью YOLO...")
    predictions = model(img_pil, conf=0.5)

    all_line_bboxes = []

    for result in predictions:
        for box in result.boxes:
            x1_block, y1_block, x2_block, y2_block = [int(i) for i in box.xyxy[0]]
            class_name = model.names[int(box.cls[0])]
            
            # Рисуем большой блок,найденный YOLO 
            draw.rectangle([x1_block, y1_block, x2_block, y2_block], outline="blue", width=2)
            draw.text((x1_block + 2, y1_block + 2), f"YOLO: {class_name}", fill="blue", font=font)

            if (x2_block - x1_block) < 15 or (y2_block - y1_block) < 10: continue
            
            cropped_img = img_pil.crop((x1_block, y1_block, x2_block, y2_block))

            #Детекция строк внутри текстовых блоков
            if class_name in ['Text', 'Title', 'List-item', 'Caption', 'Section-header', 'Footnote']:
                lines = _split_text_block_into_lines(cropped_img)
                for line_data in lines:
                    rx1, ry1, rx2, ry2 = line_data["bbox"]
                    # Пересчитываем в абсолютные координаты
                    abs_bbox = [x1_block + rx1, y1_block + ry1, x1_block + rx2, y1_block + ry2]
                    all_line_bboxes.append(abs_bbox)
    
    
    logging.info(f"Найдено {len(all_line_bboxes)} финальных строк для OCR. Рисуем их...")
    for i, bbox in enumerate(all_line_bboxes):
        draw.rectangle(bbox, outline="red", width=1)
        draw.text((bbox[0], bbox[1] - 10), str(i+1), fill="red", font=font)
        
    img_pil.save(output_path)
    logging.info(f"Готово! Изображение с результатами детекции сохранено в {output_path}")


if __name__ == '__main__':
    import logging
    import torch

    PDF_FILE_PATH = Path("./input_pdfs/analitnsu2642.pdf")
    PAGE_TO_VISUALIZE = 2 
    YOLO_MODEL = Path("./yolov8l-doclaynet.pt") 
    OUTPUT_IMAGE_PATH = Path("./output/detection_visualization.png") 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    OUTPUT_IMAGE_PATH.parent.mkdir(exist_ok=True, parents=True)

    visualize_detections(PDF_FILE_PATH, PAGE_TO_VISUALIZE, YOLO_MODEL, OUTPUT_IMAGE_PATH)