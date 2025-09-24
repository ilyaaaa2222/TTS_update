#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def json_to_structured_text(blocks: List[Dict[str, Any]]) -> str:

    if not blocks:
        return "Документ пуст."

    output_lines = []
    current_page = -1
    prev_block = None

    INDENT_THRESHOLD = 20

    for block in blocks:
        page_num = block.get("page_num", -1)
        
        if page_num != current_page:
            if current_page != -1:
                output_lines.append(f"\n\n{'='*25} Конец страницы {current_page} {'='*25}")
            current_page = page_num
            output_lines.append(f"\n\n{'='*25} Страница {page_num} {'='*25}\n")
            prev_block = None 

        block_type = block.get("type", "text")
        content = (block.get("content") or "").strip()

        if not content and block_type not in ["table", "picture"]:
            continue

        if block_type != "text":
            if content: output_lines.append(f"\n\n[Формула: {content}]\n")
            elif block_type == "table": output_lines.append("\n\n[Таблица]\n")
            elif block_type == "picture": output_lines.append("\n\n[Изображение]\n")
            prev_block = block
            continue

        if prev_block and prev_block.get("type") == "text":
            prev_bbox = prev_block["bbox"]
            curr_bbox = block["bbox"]
            
            if not all(isinstance(v, int) for v in prev_bbox + curr_bbox):
                prev_block = block
                continue

            prev_line_height = prev_bbox[3] - prev_bbox[1]
            vertical_gap = curr_bbox[1] - prev_bbox[3]
            horizontal_shift = curr_bbox[0] - prev_bbox[0]

            is_new_paragraph = (vertical_gap > prev_line_height * 0.8) or \
                               (horizontal_shift > INDENT_THRESHOLD)

            if is_new_paragraph:
                output_lines.append("\n\n" + content)
            else:
                output_lines.append(" " + content)
        else:
            output_lines.append(content)
        
        prev_block = block
        
    return "".join(output_lines).strip()

def main():
    parser = argparse.ArgumentParser(description="Преобразование JSON в структурированный TXT.")
    parser.add_argument("input_file", type=Path, help="Входной JSON файл.")
    parser.add_argument("output_file", type=Path, help="Выходной TXT файл.")
    args = parser.parse_args()

    if not args.input_file.is_file():
        print(f"Ошибка: Файл не найден: {args.input_file}")
        return

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data_blocks = json.load(f)
        
        structured_text = json_to_structured_text(data_blocks)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(structured_text)
        print(f"Готово! Результат сохранен в {args.output_file}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()