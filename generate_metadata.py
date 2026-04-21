import re
import json
from typing import Dict, Any, Optional
from PIL import Image
from google.genai import types
from google.genai.types import Tool, GoogleSearch
from google.genai.client import Client
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

thread_pool = ThreadPoolExecutor(max_workers=20)

def build_auto_data_list(obj: Optional[Dict[str, str]] = None) -> dict:
    category = "Object"
    if obj and obj.get("object_name"):
        category = obj["object_name"]

    return {
        "category": category,
        "existing_tag_values": [],
        "fields": [
            {"field_name": "Brand", "field_type": "TEXT", "occurrence_type": "SINGLE"},
            {"field_name": "Model", "field_type": "TEXT", "occurrence_type": "SINGLE"},
            {"field_name": "Color", "field_type": "TEXT", "occurrence_type": "MULTIPLE"},
            {"field_name": "Material", "field_type": "TEXT", "occurrence_type": "MULTIPLE"},
            {"field_name": "Type", "field_type": "TEXT", "occurrence_type": "SINGLE"},
            {"field_name": "Style", "field_type": "TEXT", "occurrence_type": "MULTIPLE"},
            {"field_name": "Pattern", "field_type": "TEXT", "occurrence_type": "MULTIPLE"},
            {"field_name": "Size", "field_type": "TEXT", "occurrence_type": "SINGLE"}
        ]
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON object found in response.")

    json_text = text[start:end + 1]
    json_text = re.sub(r"\bTRUE\b", "true", json_text)
    json_text = re.sub(r"\bFALSE\b", "false", json_text)
    json_text = re.sub(r",\s*}", "}", json_text)
    json_text = re.sub(r",\s*]", "]", json_text)

    return json.loads(json_text)


async def generate_metadata(
    client: Client,
    image: Image.Image,
    obj: Optional[Dict[str, str]] = None,
    search_tool: bool = False
) -> Dict[str, Any]:
    data_list = build_auto_data_list(obj)

    prompt = f"""
You are an expert asset cataloger.

Analyze the image and return only valid JSON in exactly this structure:
STRICTLY USE GOOGLE SEARCH TOOL BEFORE GENERATION

{{
  "Data": {{
    "title": "short clear product title",
    "description": "clear factual description based only on visible details",
    "tags": [
      {{
        "tagValue": "",
        "tagType": "",
        "confidenceScore": "",
        "source": "visible in image"
      }}
    ],
    "fields": [
      {{
        "field_name": "",
        "field_type": "",
        "occurrence_type": "",
        "field_value": "",
        "confidenceScore": "",
        "source": "visible in image"
      }}
    ]
  }}
}}

Rules:
- Use only visible evidence from the image.
- Do not hallucinate.
- Do not mention background, lighting, camera angle, or condition unless directly relevant to product identity.
- Keep title short and useful.
- Keep description clean and factual.
- Tags must be simple and useful.
- Fill fields only when reasonably visible.
- If a value is not visible, leave it empty.
- Return JSON only.
- No markdown.
- No extra text.

Use this category and field guidance:
{json.dumps(data_list, ensure_ascii=False)}
"""

    config = types.GenerateContentConfig(
        temperature=0.0,
        response_modalities=["TEXT"],
    )

    if search_tool:
        google_search_tool = Tool(google_search=GoogleSearch())
        config = types.GenerateContentConfig(
            temperature=0.02,
            seed=42,
            response_modalities=["TEXT"],
            tools=[google_search_tool],
            # thinking_config=types.ThinkingConfig(thinking_budget=512)
        )
    start_time_metadata_extract = time.time()
    loop = asyncio.get_running_loop()

    input_token_count = 0
    cache_token_count = 0
    output_token_count = 0

    response = await asyncio.wait_for(
        loop.run_in_executor(
            thread_pool,
            lambda: client.models.generate_content(
            # model="gemini-3.1-flash-lite-preview",
            model="gemini-2.5-flash-lite",
            contents=[image, prompt],
            config=config
            )
        ),
        timeout=40
    )


    end_time_metadata_extract = time.time() - start_time_metadata_extract
    print(response)
    metadata = response.usage_metadata
    # input_token_count = 0
    # cache_token_count = 0
    # output_token_count = 0

    if metadata:
        output_token_count = metadata.candidates_token_count or 0
        prompt_token_count = metadata.prompt_token_count or 0
        cache_token_count = metadata.cached_content_token_count or 0
        input_token_count = prompt_token_count - cache_token_count

    if response.candidates and response.candidates[0].grounding_metadata:
        search_entry_point = response.candidates[0].grounding_metadata.search_entry_point
    else:
        search_entry_point = None

    search_tool_used = "Yes" if search_entry_point else "No"

    json_res = _extract_json_object(response.text)

    data = json_res.get("Data", {})
    return {
        "Title": data.get("title", ""),
        "Description": data.get("description", ""),
        "Tags": data.get("tags", []),
        "Fields": data.get("fields", []),
        "Json Response": json_res,
        "Input Token Count": input_token_count,
        "Cached Token Count": cache_token_count,
        "Output Token Count": output_token_count,
        "Time Taken": end_time_metadata_extract,
        "Search Tool Used": search_tool_used,
        "Cache Used": "No" if cache_token_count == 0 else "Yes"
    }

# import requests
# import base64
# import json
# import time
# import re
# from PIL import Image
# import os
# from dotenv import load_dotenv

# dir = os.getcwd()
# ENV_PATH = os.path.join(dir, ".env")
# load_dotenv(ENV_PATH)
# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# def encode_image(image: Image.Image):
#     import io
#     buffer = io.BytesIO()
#     image.save(buffer, format="JPEG", quality=70)
#     return base64.b64encode(buffer.getvalue()).decode()


# async def generate_metadata_fast(image, obj):
#     data_list = build_auto_data_list(obj)

#     base64_img = encode_image(image)

#     prompt = f"""
# You are an expert asset cataloger.

# Analyze the image and return only valid JSON in exactly this structure:

# {{
#   "Data": {{
#     "title": "short clear product title",
#     "description": "clear factual description based only on visible details",
#     "tags": [
#       {{
#         "tagValue": "",
#         "tagType": "",
#         "confidenceScore": "",
#         "source": "visible in image"
#       }}
#     ],
#     "fields": [
#       {{
#         "field_name": "",
#         "field_type": "",
#         "occurrence_type": "",
#         "field_value": "",
#         "confidenceScore": "",
#         "source": "visible in image"
#       }}
#     ]
#   }}
# }}

# Rules:
# - Use only visible evidence from the image.
# - Do not hallucinate.
# - Do not mention background, lighting, camera angle, or condition unless directly relevant to product identity.
# - Keep title short and useful.
# - Keep description clean and factual.
# - Tags must be simple and useful.
# - Fill fields only when reasonably visible.
# - If a value is not visible, leave it empty.
# - Return JSON only.
# - No markdown.
# - No extra text.

# Use this category and field guidance:
# {json.dumps(data_list, ensure_ascii=False)}
# """
#     def sync_call():
#         return requests.post(
#             url="https://openrouter.ai/api/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#                 "Content-Type": "application/json",
#             },
#             json={
#                 "model": "x-ai/grok-4.1-fast",
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": prompt},
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/jpeg;base64,{base64_img}"
#                                 }
#                             }
#                         ]
#                     }
#                 ],
#                 "temperature": 0.3,
#                 "max_tokens": 1000,
#                 "reasoning": {"enabled": False}
#             }
#         )
#     start = time.time()

#     response = await asyncio.to_thread(sync_call)

#     res = response.json()

#     output = res["choices"][0]["message"]["content"].strip()

#     # 🔥 CLEAN JSON
#     if output.startswith("```"):
#         output = re.sub(r"^```(?:json)?\s*", "", output)
#         output = re.sub(r"\s*```$", "", output)

#     try:
#         data = json.loads(output)
#     except:
#         match = re.search(r"\{.*\}", output, re.DOTALL)
#         if not match:
#             raise ValueError("Invalid JSON from Grok metadata")
#         data = json.loads(match.group(0))

#     end = time.time() - start

#     # 🔥 ADD PIPELINE METRICS (important for your UI)
#     data["Time Taken"] = end
#     data["Input Token Count"] = res.get("usage", {}).get("prompt_tokens", 0)
#     data["Output Token Count"] = res.get("usage", {}).get("completion_tokens", 0)
#     data["Search Tool Used"] = False

#     data_block = data.get("Data", {})

#     return {
#         "Title": data_block.get("title", ""),
#         "Description": data_block.get("description", ""),
#         "Tags": data_block.get("tags", []),
#         "Fields": data_block.get("fields", []),
#         "Json Response": data,
#         "Input Token Count": res.get("usage", {}).get("prompt_tokens", 0),
#         "Output Token Count": res.get("usage", {}).get("completion_tokens", 0),
#         "Time Taken": end,
#         "Search Tool Used": "No"
#     }