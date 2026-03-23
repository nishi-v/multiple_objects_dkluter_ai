import os
import re
import json
from typing import Dict, Any, Optional
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Tool, GoogleSearch
from google.genai.client import Client

def init_client() -> Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or .env")
    return genai.Client(api_key=api_key)


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


def generate_metadata(
    image: Image.Image,
    obj: Optional[Dict[str, str]] = None,
    search_tool: bool = False
) -> Dict[str, Any]:
    client = init_client()
    data_list = build_auto_data_list(obj)

    prompt = f"""
You are an expert asset cataloger.

Analyze the image and return only valid JSON in exactly this structure:

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
            temperature=0.0,
            response_modalities=["TEXT"],
            tools=[google_search_tool],
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[image, prompt],
        config=config
    )

    usage = response.usage_metadata
    input_token_count = 0
    cache_token_count = 0
    output_token_count = 0

    if usage:
        output_token_count = usage.candidates_token_count or 0
        prompt_token_count = usage.prompt_token_count or 0
        cache_token_count = usage.cached_content_token_count or 0
        input_token_count = prompt_token_count - cache_token_count

    parsed = _extract_json_object(response.text)

    data = parsed.get("Data", {})
    return {
        "Title": data.get("title", ""),
        "Description": data.get("description", ""),
        "Tags": data.get("tags", []),
        "Fields": data.get("fields", []),
        "Json Response": parsed,
        "Input Token Count": input_token_count,
        "Cached Token Count": cache_token_count,
        "Output Token Count": output_token_count,
        "Search Tool Used": "Yes" if search_tool else "No",
        "Cache Used": "No"
    }