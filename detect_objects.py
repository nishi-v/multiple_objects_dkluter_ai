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

# dir = Path(os.getcwd())

# # Load environment variables from .env file
# ENV_PATH :Path= dir / '.env'

# load_dotenv(ENV_PATH)
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise ValueError("GEMINI_API_KEY not found in environment or .env")
# client = genai.Client(api_key=api_key)
def resize_image(image: Image.Image, max_side: int = 768) -> Image.Image:
    width, height = image.size

    # already small enough
    if max(width, height) <= max_side:
        return image

    scale = max_side / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.LANCZOS)

async def detect_objects(client: Client, image: Image.Image) -> tuple[List[Dict[str, str]], float]:
    prompt = """
Analyze the original input image and detect the main visible objects.

Rules:
- Detect objects only from the original input image.
- Use only clear visible evidence from the image.
- object_name must correctly match the visible object.
- short_description must contain only visible details of that same object.
- Do not guess, assume, or hallucinate.
- If exact identity is unclear, use a simple generic name that still matches the visible object.

- Detect each real object only once.
- Do not duplicate detections.
- Do not split one real object into multiple detections if it visually reads as one object.
- If multiple same-type items clearly appear as one group, detect them as one object group.
- If items are placed inside, inserted into, stored in, or held by a holder, organizer, stand, tray, rack, box, cup, pot, or container, and together they visually read as one setup, detect them as one object.
- If the holder/container and its contents are clearly being shown as one combined object, keep them together.
- Separate objects only when they clearly appear as independent standalone objects.
- If one object is simply placed on top of another but they still clearly look separate, detect them separately.

- Look carefully at small objects, similar-looking objects, and partially visible objects before naming them.
- Do not detect hidden or non-visible objects.
- Ignore background, shadows, reflections, and support surfaces.

Output:
- Return only a valid JSON array.
- No markdown.
- No extra text.
- Each item must have:
  object_name, object_id, short_description, position_hint
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
                model="gemini-2.5-flash-lite",
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.02,
                    # tools=[google_search_tool]
                )
            )
        ),
        timeout=40
    )
    end_time_detect_objects = time.time() - start_time_detect_objects

    response_text = response.text.strip()

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
        position_hint = str(obj.get("position_hint", "")).strip()

        if not object_name:
            continue

        if object_name.lower() in blocked:
            continue

        key = (
            object_name.lower(),
            short_description.lower(),
            position_hint.lower()
        )
        if key in seen:
            continue
        seen.add(key)

        objects.append({
            "object_name": object_name,
            "object_id": f"object_{len(objects)+1}",
            "short_description": short_description,
            "position_hint": position_hint
        })

    return objects, end_time_detect_objects