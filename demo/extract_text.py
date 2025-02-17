import re
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
from magic_pdf.data.dataset import ImageDataset
import os
import json
import cv2
import numpy as np

def get_image_size_from_result(result):
    """从识别结果中获取图片尺寸"""
    max_width = 0
    max_height = 0
    for det in result:
        if det.get("poly"):
            poly = det["poly"]
            # poly格式为 [x1,y1,x2,y2,x3,y3,x4,y4]
            for i in range(0, len(poly), 2):
                max_width = max(max_width, poly[i])
                max_height = max(max_height, poly[i+1])
    return int(max_width), int(max_height)

def extract_text_from_result(result):
    """从识别结果中提取文本内容"""
    # 按y坐标对所有元素进行排序
    items = []
    for det in result:
        if det.get("poly"):
            y_center = (det["poly"][1] + det["poly"][5]) / 2
            x_center = (det["poly"][0] + det["poly"][2]) / 2
            items.append((y_center, x_center, det))
    
    items.sort(key=lambda x: (x[0], x[1]))  # 先按y坐标，再按x坐标排序
    
    content = []
    
    # 处理所有识别到的文本
    for y, x, det in items:
        if det.get("text"):
            content.append(det["text"])
        elif det.get("html"):
            content.append(det["html"])
    
    # 返回识别的内容
    return "\n".join(content)

def main():
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化模型
    model_singleton = ModelSingleton()
    custom_model = model_singleton.get_model(ocr=True, show_log=True, lang="ch")
    
    # 读取并处理图片
    image_path = "image1.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
        
    height, width = image.shape[:2]  # 获取图片尺寸
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    
    # 使用模型处理图片
    result = custom_model(image)
    
    # 从结果中获取实际使用的图片尺寸
    model_width, model_height = get_image_size_from_result(result)
    
    # 保存模型结果
    model_json_path = os.path.join(output_dir, "result_model.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 添加调试信息：检查result_model.json的内容
    print("\n检查保存的JSON文件内容...")
    with open(model_json_path, "r", encoding="utf-8") as f:
        saved_result = json.load(f)
        print(f"JSON文件中的数据条数: {len(saved_result)}")
        print("检查category_id为5的内容:")
        for item in saved_result:
            if item.get("category_id") == 5:
                print(f"找到表格数据")
    
    # 创建数据集
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    dataset = ImageDataset(image_bytes)
    
    # 使用MagicModel处理
    magic_model = MagicModel([{
        "layout_dets": result,
        "page_info": {
            "page_no": 0,
            "width": model_width,
            "height": model_height
        }
    }], dataset)
    
    # 生成markdown文件
    markdown_path = os.path.join(output_dir, "result.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        # 获取完整的文本内容
        text_content = extract_text_from_result(saved_result)  # 使用保存的结果
        f.write(text_content)

if __name__ == "__main__":
    main()
