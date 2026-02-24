import google.generativeai as genai
import os
from PIL import Image
import time
from openai import OpenAI
import base64
import io

def process_image_with_gemini(image_path, api_key, model_name='gemini-1.5-flash-latest'):
    """
    使用 Gemini API 對單張圖片進行 OCR。
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        with Image.open(image_path) as img:
            prompt = "你是一個專業的 OCR 引擎。請將這張圖片中的所有內容，包含標題、段落、列表和表格，轉換為結構良好、語法正確的 Markdown 格式。請盡力還原原始的排版結構。"
            response = model.generate_content([prompt, img])

        if response.parts:
            markdown_text = response.parts[0].text
            print(f"成功使用 Gemini 處理圖片 {image_path}。")
            return markdown_text
        else:
            print(f"處理圖片 {image_path} 後，Gemini 未回傳有效內容。")
            return ""

    except Exception as e:
        print(f"呼叫 Gemini API 處理圖片 {image_path} 時發生錯誤: {e}")
        return None

def process_image_with_openai(image_path, api_key, base_url, model_name):
    """
    使用 OpenAI 相容 API 對單張圖片進行 OCR。
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = "你是一個專業的 OCR 引擎。請將這張圖片中的所有內容，包含標題、段落、列表和表格，轉換為結構良好、語法正確的 Markdown 格式。請盡力還原原始的排版結構。"
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )

        markdown_text = response.choices[0].message.content
        print(f"成功使用 OpenAI 相容 API 處理圖片 {image_path}。")
        return markdown_text

    except Exception as e:
        print(f"呼叫 OpenAI 相容 API 處理圖片 {image_path} 時發生錯誤: {e}")
        return None