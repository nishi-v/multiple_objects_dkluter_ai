import re
import json
from typing import List, Dict
from PIL import Image
from google.genai import types
from google.genai.client import Client
from google.genai.types import Tool, GoogleSearch
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

thread_pool = ThreadPoolExecutor(max_workers=20)

def resize_image(image: Image.Image, max_side: int = 1024) -> Image.Image:
    width, height = image.size

    # already small enough
    if max(width, height) <= max_side:
        return image

    scale = max_side / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.LANCZOS)

def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

def dedupe_key(obj: Dict[str, str]) -> tuple:
    return (
        normalize_text(obj.get("category", "")),
        normalize_text(obj.get("object_name", "")),
        normalize_text(obj.get("position_hint", ""))[:40],
    )

async def detect_objects(client: Client, image: Image.Image) -> tuple[List[Dict[str, str]], float]:
    prompt = """
You are an object detection model.

Task:
- Detect ALL objects present in the image.
- Include objects even if partially visible, small, folded, overlapping, or occluded.
- Do NOT miss objects.

For each object return:
- object_name
- category
- object_id
- short_description
- position_hint
- confidence_score
- visibility_score

Detection rules:
- Scan the FULL image (center, edges, corners).
- Detect every distinct physical item.
- If multiple similar objects exist, detect each separately.
- Each detected object MUST be unique in position.
- Do NOT return the same object multiple times.
- NEVER mix attributes (text, color, pattern) between different objects
- Each detected object must strictly correspond to one physical item

Object naming:
- Identify object using BOTH:
  - visible text/content
  - position in image

CRITICAL:
- When multiple similar objects exist (e.g., books):
  - DO NOT mix titles between them
  - DO NOT assign text from one object to another
  - Each object must keep its own visible text

Text rule:
- If text is visible → use ONLY that object's text
- Do NOT guess from nearby objects
- Do NOT autocomplete or replace text

Uncertainty:
- If text is partially visible:
  - use only visible portion OR generic name (e.g., "book")
  - DO NOT guess full title incorrectly

Disambiguation:
- Use position + color + layout to map correct name to correct object

Search usage:
- Use search ONLY when:
  - text, logo, or identifiable pattern is visible
  - object appears recognizable but incomplete
- Use search to CONFIRM, not to override.
- If search conflicts with image → follow image.

Disambiguation:
- Use position, color, pattern, size to distinguish similar objects.
- Ensure each object has correct identity based on its own visual features.

Clothing rule:
- If full structure visible → name specific (e.g., pants, shirt).
- If folded/partial → use "garment" or "folded clothing".

Pair / group rule:
- If objects naturally come in pairs (shoes, earrings):
  - Detect as ONE object if both are together.
  - If only one visible → detect as single item.

Separation rules:
- Separate objects that are visually distinct.
- Do NOT merge different items.
- Do NOT detect the same object twice.

Dense scene rule:
- Carefully detect small, partially hidden, and overlapping objects.
- Do NOT stop after detecting a few objects.

Position:
- Use simple location (top left, center, bottom right, etc).

Scores:
- confidence_score → how correct the label is
- visibility_score → how visible the object is
- Lower scores if object is unclear or occluded

Critical rules:
- Prefer detecting more objects rather than missing them.
- Do NOT skip objects due to uncertainty.
- If unsure → include with lower confidence and generic name.

Output:
- Return ONLY a valid JSON array
- No markdown
- No extra text
"""

    google_search_tool = Tool(
        google_search=GoogleSearch(),
    )
    start_time_detect_objects = time.time()
    loop = asyncio.get_running_loop()

    response = await asyncio.wait_for(
        loop.run_in_executor(
            thread_pool,
            lambda: client.models.generate_content(
                # model="gemini-3.1-flash-lite-preview",
                model="gemini-2.5-flash-lite",
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.02,
                    # thinking_config=types.ThinkingConfig(thinking_budget=512),
                    tools=[google_search_tool]
                )
            )
        ),
        timeout=40
    )
    end_time_detect_objects = time.time() - start_time_detect_objects

    try:
        raw_text = getattr(response, "text", None)
        if raw_text is None:
            raise ValueError("No text returned by model.")

        response_text = raw_text.strip()
        if not response_text:
            raise ValueError("Empty text returned by model.")
    except Exception as e:
        raise ValueError(f"Failed to read model response text: {e}")

    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\[\s*{.*}\s*\]", response_text, re.DOTALL)
        if not match:
            raise ValueError("Model did not return valid JSON array text.")
        data = json.loads(match.group(0))

    blocked = {
        "shelf", "shelves", "desk", "table", "floor", "wall",
        "background", "furniture", "surface", "counter", "rack"
    }

    objects = []
    seen = set()

    for obj in data:
        object_name = str(obj.get("object_name", "")).strip()
        short_description = str(obj.get("short_description", object_name)).strip()
        # visible_description = str(obj.get("visible_description", object_name)).strip()
        position_hint = str(obj.get("position_hint", "")).strip()
        confidence_score = str(obj.get("confidence_score", "")).strip()
        visibility_score = str(obj.get("visibility_score", "")).strip()

        if not object_name:
            continue

        if object_name.lower() in blocked:
            continue

        key = dedupe_key(obj)

        if key in seen:
            continue

        seen.add(key)

        category = str(obj.get("category", "")).strip().lower()
        if not category:
            parts = normalize_text(object_name).split()
            category = parts[-1] if parts else normalize_text(object_name)


        objects.append({
            "object_name": object_name,
            "category": category,
            "object_id": f"object_{len(objects)+1}",
            "short_description": short_description,
            # "visible_description": visible_description,
            "position_hint": position_hint,
            "confidence_score": confidence_score,
            "visibility_score": visibility_score
        })


    return objects, end_time_detect_objects

# import base64
# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# import requests
# dir = os.getcwd()
# ENV_PATH = os.path.join(dir, ".env")
# load_dotenv(ENV_PATH)
# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# # client = OpenAI(
# #     api_key=OPENROUTER_API_KEY,
# #     base_url="https://openrouter.ai/api/v1"
# # )

# def encode_image(image):
#     import io
#     buffer = io.BytesIO()
#     image.save(buffer, format="JPEG", quality=50)  # compress = faster
#     return base64.b64encode(buffer.getvalue()).decode()

# async def detect_objects_fast(image):

#     base64_img = encode_image(image)

#     prompt = """
#  Analyze the original input image and detect the main visible objects.

#  Rules:
#  - Detect objects only from the original input image.
#  - Use only clear visible evidence from the image.
#  - object_name must correctly match the visible object.
#  - short_description must contain only visible details of that same object.
#  - Do not guess, assume, or hallucinate.
#  - If exact identity is unclear, use a simple generic name that still matches the visible object.

#  - Detect each real object only once.
#  - Do not duplicate detections.
#  - Do not split one real object into multiple detections if it visually reads as one object.
#  - If multiple same-type items clearly appear as one group, detect them as one object group.
#  - If items are placed inside, inserted into, stored in, or held by a holder, organizer, stand, tray, rack, box, cup, pot, or container, and together they visually read as one setup, detect them as one object.
#  - If the holder/container and its contents are clearly being shown as one combined object, keep them together.
#  - Separate objects only when they clearly appear as independent standalone objects.
#  - If one object is simply placed on top of another but they still clearly look separate, detect them separately.

#  - Look carefully at small objects, similar-looking objects, and partially visible objects before naming them.
#  - Do not detect hidden or non-visible objects.
#  - Ignore background, shadows, reflections, and support surfaces.

#  Output:
#  - Return only a valid JSON array.
#  - No markdown.
#  - No extra text.
#  - Each item must have:
#    object_name, object_id, short_description, position_hint
#  """

#     start = time.time()

#     response = requests.post(
#         url="https://openrouter.ai/api/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#             "Content-Type": "application/json",
#         },
#         json={  # ✅ use json= instead of data=
#             "model": "x-ai/grok-4.1-fast",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{base64_img}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             "temperature": 0.0,              # 🔥 deterministic
#             "max_tokens": 500,
#             "reasoning": {"enabled": False}  # 🔥 faster
#         }
#     )

#     res = response.json()

#     output = res["choices"][0]["message"]["content"].strip()

#     # 🔥 CLEAN RESPONSE
#     if output.startswith("```"):
#         output = re.sub(r"^```(?:json)?\s*", "", output)
#         output = re.sub(r"\s*```$", "", output)

#     try:
#         data = json.loads(output)
#     except:
#         match = re.search(r"\[\s*{.*}\s*\]", output, re.DOTALL)
#         if not match:
#             raise ValueError("Invalid JSON")
#         data = json.loads(match.group(0))

#     # 🔥 NORMALIZE (match your app format)
#     objects = []
#     for i, obj in enumerate(data):
#         objects.append({
#             "object_name": obj.get("object_name", ""),
#             "object_id": f"object_{i+1}",
#             "short_description": obj.get("short_description", ""),
#             "position_hint": obj.get("position_hint", "")
#         })

#     end = time.time() - start

#     return objects, end