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
You are an expert universal visual object detection and identification model.

Task:
Detect the MAXIMUM number of real visible physical objects in the image and return ONLY a valid JSON array.

PRIMARY GOALS:
1. Detect as many real objects as possible.
2. Include full, partial, cropped, overlapping, stacked, small, edge, and background objects when supported by visible evidence.
3. Use Google Search to recover missing identity/details from visible clues.
4. Prevent fake detections.
5. Prevent duplicates.
6. Never hallucinate.

DETECTION PRIORITY:
- High recall is important.
- Do NOT skip obvious or likely real objects.
- If an object is reasonably supported by visible evidence, include it.
- Use lower confidence for uncertain or partial objects instead of skipping.

CORE OBJECT RULE:
Detect an object when there is reasonable evidence of a separate physical item.

Evidence may include:
- edges
- corners
- surface area
- shape
- thickness
- texture
- material
- readable text
- logo
- label
- packaging
- handle
- wheel
- strap
- pages
- spine
- buttons
- ports
- distinct placement

STRICT:
- Text alone should not create a fake object.
- Search alone should not create a fake object.
- Reflection/shadow/pattern is not an object.

PARTIAL OBJECT RULE:
- Full visibility is NOT required.
- Detect partially visible objects.
- Detect cropped border objects.
- Detect hidden objects when enough visible clues exist.
- Detect stacked and overlapping objects separately when evidence exists.
- If only part is visible, still include it.

GOOGLE SEARCH RULE (IMPORTANT):
Use Google Search aggressively but correctly when visible clues exist:

Use search for:
- readable text
- partial readable text
- logo
- brand mark
- packaging style
- model number
- title
- author name
- distinctive product design
- electronics design
- known product shapes

Search can help recover:
- missing names
- missing brands
- missing book titles
- missing product identity
- likely model/category

Search must support visible evidence.
Search must not invent fake objects.

TEXT RULE:
- Use text on same object only.
- Never borrow distant text.
- Partial text may be used with search if physically attached to object.

GLOBAL SCAN RULE:
Carefully inspect:
- top
- bottom
- left
- right
- center
- corners
- foreground
- background
- shelves
- stacks
- piles
- under overlaps

Detect all categories:
clothes, books, electronics, decor, toys, tools, bags, footwear, boxes, bottles, cosmetics, furniture items, stationery, appliances, sports items, containers, gadgets, plants, vehicles, pair of shoes, etc.

DUPLICATE RULE:
- Detect each object once only.
- Separate identical objects if physically separate.

NAMING RULE:
- If search strongly confirms identity, use specific name.
- If not certain, use generic correct name.
- Better to include generic real object than skip it.

VISIBLE_DESCRIPTION RULE:
Describe visible reality:
- color
- material
- texture
- shape
- logo
- text
- graphics
- pattern
- design
- wear
- orientation
- visible portion

POSITION VALUES:
top left
top center
top right
center left
center
center right
bottom left
bottom center
bottom right

OUTPUT:
Return ONLY valid JSON array.

Each item:
{
  "object_name": "",
  "category": "",
  "short_description": "",
  "visible_description": "",
  "position_hint": "",
  "confidence_score": "",
  "visibility_score": ""
}

FIELD RULES:
- confidence_score = confidence object is correct (0-100)
- visibility_score = how visible object is (0-100)

FINAL RULES:
- Missing obvious objects is bad.
- Include supported partial objects.
- Use search to fill missing info.
- No markdown.
- No explanation.
- No extra text.
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
                model="gemini-2.5-flash",
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    # thinking_config=types.ThinkingConfig(thinking_budget=512),
                    tools=[google_search_tool]
                )
            )
        ),
        timeout=50
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

    if response.candidates and response.candidates[0].grounding_metadata:
        search_entry_point = response.candidates[0].grounding_metadata.search_entry_point
    else:
        search_entry_point = None

    search_tool_used = "Yes" if search_entry_point else "No"

    blocked = {
        "shelf", "shelves", "desk", "table", "floor", "wall",
        "background", "furniture", "surface", "counter", "rack"
    }

    objects = []
    seen = set()

    for obj in data:
        object_name = str(obj.get("object_name", "")).strip()
        short_description = str(obj.get("short_description", object_name)).strip()
        visible_description = str(obj.get("visible_description", object_name)).strip()
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
            "visible_description": visible_description,
            "position_hint": position_hint,
            "confidence_score": confidence_score,
            "visibility_score": visibility_score,
        })


    return objects, end_time_detect_objects, search_tool_used
