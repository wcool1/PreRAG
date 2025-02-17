# Copyright (c) Opendatalab. All rights reserved.
import os
import json
import re
import cv2
import ollama
from pathlib import Path
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze, ModelSingleton
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.read_api import read_local_office, read_local_images
from extract_text import extract_text_from_result

def get_image_description(image_path):
    """
    使用LLM获取图片描述
    Args:
        image_path: 图片路径
    Returns:
        str: 图片描述文本
    """
    try:
        response = ollama.chat(
            model='minicpm-v',
            messages=[{
                'role': 'user',
                'content': '请描述这张图片的内容（特征和细节），使用中文回答，要求详细和全面',
                'images': [image_path]
            }]
        )
        return response['message']['content']
    except Exception as e:
        print(f"LLM处理图片时出错: {str(e)}")
        return f"[图片描述生成失败: {str(e)}]"

def process_file(file_path, output_base_dir):
    """
    处理单个文件（支持PDF、Office文档和图片）
    Args:
        file_path: 输入文件路径
        output_base_dir: 输出基础目录
    """
    # 获取文件名和扩展名
    file_name = os.path.basename(file_path)
    name_without_suff, file_ext = os.path.splitext(file_name)
    file_ext = file_ext.lower()

    # 准备输出环境：临时目录用于存储中间文件
    temp_dir = os.path.join(output_base_dir, "_temp", name_without_suff)
    temp_image_dir = os.path.join(temp_dir, "images")
    os.makedirs(temp_image_dir, exist_ok=True)

    # 初始化文件写入器
    image_writer = FileBasedDataWriter(temp_image_dir)
    md_writer = FileBasedDataWriter(temp_dir)

    # 根据文件类型选择处理方式
    if file_ext in ['.pdf']:
        # PDF文件处理
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(str(file_path))
        ds = PymuDocDataset(pdf_bytes)
        
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

    elif file_ext in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
        # Office文档处理
        ds = read_local_office(file_path)[0]
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        # 图片文件处理
        ds = read_local_images(file_path)[0]
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

    else:
        raise ValueError(f"不支持的文件类型: {file_ext}")

    # 生成临时markdown文件
    temp_md_path = os.path.join(temp_dir, f"{name_without_suff}.md")
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", "images")

    # 处理markdown中的图片
    if os.path.exists(temp_md_path):
        with open(temp_md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # 修改正则表达式以准确匹配markdown图片语法
        image_pattern = f"!\\[(.*?)\\]\\(images/([^)]+)\\)"
        
        # 使用列表存储所有需要处理的图片
        images_to_process = list(re.finditer(image_pattern, md_content))
        if images_to_process:
            print(f"找到 {len(images_to_process)} 个图片需要处理")
            
            model_singleton = ModelSingleton()
            custom_model = model_singleton.get_model(ocr=True, show_log=True, lang="ch")
            
            # 从后向前处理，避免替换位置变化影响
            for match in reversed(images_to_process):
                image_name = match.group(2)  # 使用group(2)获取文件名
                image_path = os.path.join(temp_image_dir, image_name)
                print(f"处理图片: {image_path}")
                
                image = cv2.imread(image_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 进行OCR识别
                    result = custom_model(image_rgb)
                    ocr_text = extract_text_from_result(result)
                    
                    # 使用LLM生成图片描述
                    llm_description = get_image_description(image_path)
                    
                    # 检查OCR文本是否为空
                    if not ocr_text.strip():
                        ocr_text = "该图片中未识别到文字内容"
                    
                    # 组合OCR文本和LLM描述，调整顺序和格式
                    combined_text = f"""

- 图片描述：

{llm_description}

-------------------

- 图片内容：

{ocr_text}


                        """
                    
                    # 获取完整的匹配文本并替换
                    full_match = match.group(0)
                    print(f"替换图片链接: {full_match}")
                    md_content = md_content.replace(full_match, combined_text)
                else:
                    print(f"无法读取图片: {image_path}")
        
        # 保存最终的markdown文件到output目录
        final_md_path = os.path.join(output_base_dir, f"{name_without_suff}.md")
        with open(final_md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"处理完成，已保存到: {final_md_path}")

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

def process_directory(input_dir, output_dir="output"):
    """
    处理指定目录下的所有支持的文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，默认为'output'
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"输入目录 {input_dir} 不存在")
        return
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的文件扩展名
    supported_extensions = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'
    }
    
    # 获取所有支持的文件
    files = []
    for ext in supported_extensions:
        files.extend(list(input_path.glob(f"**/*{ext}")))
    
    total_files = len(files)
    print(f"找到 {total_files} 个支持的文件")
    
    for index, file in enumerate(files, 1):
        print(f"正在处理 [{index}/{total_files}]: {file}")
        try:
            process_file(file, output_path)
            print(f"成功处理: {file}")
        except Exception as e:
            print(f"处理 {file} 时出错: {str(e)}")
    
    print("处理完成！")

if __name__ == "__main__":
    import sys

    # 设置默认输入路径和输出目录
    input_path = "./resource"
    output_dir = "output"

    # 如果命令行提供了参数，则使用命令行参数
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    # 将输入路径转换为Path对象
    input_path = Path(input_path)
    
    # 检查输入路径是否存在
    if not input_path.exists():
        print(f"错误：输入路径 {input_path} 不存在")
        sys.exit(1)
    
    # 根据输入路径类型选择相应的处理方式
    if input_path.is_file():
        # 检查文件扩展名
        file_ext = input_path.suffix.lower()
        supported_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'
        }
        
        if file_ext in supported_extensions:
            print(f"正在处理文件: {input_path}")
            try:
                process_file(input_path, output_dir)
                print(f"成功处理文件: {input_path}")
            except Exception as e:
                print(f"处理文件时出错: {str(e)}")
        else:
            print(f"错误：不支持的文件类型 {file_ext}")
    elif input_path.is_dir():
        print(f"正在处理目录: {input_path}")
        process_directory(input_path, output_dir)
    else:
        print(f"错误：输入路径 {input_path} 既不是文件也不是目录")
