import os
import re
from typing import Dict
from PIL import Image
from google.genai import types
from google.genai.client import Client
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

thread_pool = ThreadPoolExecutor(max_workers=20)

# def init_client(env_path:Union[Path, str]) -> Client:
#     load_dotenv(env_path)
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise ValueError("GEMINI_API_KEY not found in environment or .env")
#     return genai.Client(api_key=api_key)

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")

async def generate_object_image(
    client: Client,
    reference_image: Image.Image,
    obj: Dict[str, str],
    output_path: str
) ->float:
    prompt = f"""
Task:
- Generate ONLY the selected object from the input image.

Input:
- object_name: {obj['object_name']}
- category: {obj.get('category', obj['object_name'])}
- location: {obj.get('position_hint', '')}

Core rule:
- First LOCATE the exact object using object_name + location.
- Then USE that SAME object and COMPLETE it.
- This is NOT full image generation.
- This is PARTIAL EXTENSION of an existing object.

Object selection (VERY IMPORTANT):
- If multiple similar objects exist:
  - Use location + color + pattern + size to identify correct instance.
- MUST match exact object from image.
- Do NOT switch to another similar object.
- Do NOT reuse same object for different inputs.

CRITICAL — Visible part (STRICT LOCK):
- The visible part is FINAL and must NOT be changed.

ZERO CHANGE allowed:
- color (no correction, no shift)
- texture (no smoothing or sharpening)
- material
- pattern / design
- text / font / layout
- folds / edges / shape

STRICT:
- Do NOT redraw it
- Do NOT recreate it
- Do NOT enhance or fix it
- Do NOT rotate, align, or correct it
- Treat it as FIXED and UNTOUCHABLE

CRITICAL — Graphic/Text Lock:
- Any printed graphics (book covers, labels, artwork) are PART OF VISIBLE REGION.
- These must NOT be regenerated.

STRICT:
- Do NOT redraw cover design
- Do NOT recreate typography
- Do NOT approximate layout
- Do NOT restyle graphics

- Treat cover/label as an image patch and KEEP IT EXACT.

NO STYLE CORRECTION:
- Do NOT make it cleaner, sharper, brighter, or more realistic
- Do NOT normalize colors or lighting
- Keep original imperfections exactly

Completion rule:
- Generate ONLY missing (non-visible) parts
- Extend outward from visible edges
- Do NOT rebuild the object

Completion must follow:
- same color
- same texture
- same material
- same pattern/design

Think:
- "continue the same object outward"
- NOT "generate a new version"

Text rule:
- If object contains text:
  - keep visible text EXACTLY same
  - do NOT change font or layout
  - do NOT replace with other text
  - do NOT hallucinate missing text

Anti-mixing (CRITICAL):
- Do NOT mix features from other objects
- Do NOT borrow colors, text, or patterns from nearby items
- Do NOT create hybrid objects

Identity rule:
- Output must be SAME object instance
- Not similar, not improved, not replaced

Strict constraints:
- No redesign
- No style change
- No structure change
- Visible region = ZERO modification

Output:
- One object only
- Fully completed
- Clean plain background
- No extra objects
"""

    start_time_gen_obj = time.time()
    loop = asyncio.get_running_loop()

    response = await asyncio.wait_for(
        loop.run_in_executor(
            thread_pool,
            lambda: client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                # model="gemini-2.5-flash-lite",
                contents=[prompt, reference_image],
                config=types.GenerateContentConfig(
                    # service_tier=,
                    temperature=0.02,
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(image_size="512", aspect_ratio="1:1"),
                    # thinking_config=types.ThinkingConfig(thinking_level="low"),
                    # thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
        ),
        timeout = 180    
    )
    end_time_gen_obj = time.time() - start_time_gen_obj

        # Guard against None response
    if not response or not response.candidates:
        raise ValueError(f"No candidates returned for {obj['object_id']}")

    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and inline_data.data:
                with open(output_path, "wb") as f:
                    f.write(inline_data.data)
                return end_time_gen_obj

    raise ValueError(f"No image data found in response for {obj['object_id']}")
