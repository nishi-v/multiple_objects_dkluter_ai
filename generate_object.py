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
Use the original input image to identify the target object.

Target object:
- type: {obj['object_name']}
- description: {obj['short_description']}
- location: {obj['position_hint']}

Generate only the target object from the original input image.

Rules:
- The original input image is the only source of truth.
- Use the text only to help locate the object in the image.
- If the text is wrong, incomplete, or slightly misleading, follow the actual visible object in the original input image.
- Generate the exact same real object or exact same visible object group from the original input image.
- Keep the exact same visible identity.
- Keep the exact same visible color, shape, structure, material, texture, parts, details, and quantity.
- Keep the same visible arrangement, grouping, overlap, spacing, proportions, and orientation.
- Do not add anything.
- Do not remove anything.
- Do not replace anything.
- Do not change anything.
- Do not hallucinate any new part, feature, detail, hidden part, or extra element.
- Do not include nearby, overlapping, top, bottom, inside, supporting, or background objects unless they are clearly part of the same target object.
- If another separate object is touching, covering, overlapping, or stacked on the target, do not include it.
- If part of the target is unclear or cropped, continue only the same object in the safest minimal way without inventing new visible design details.
- Keep the result looking like the same object from the original input image, not a similar object.
- Keep the result in a simple front view.
- The full target object or full target group must be fully visible.
- No cropping, no extra objects, no text, no watermark.
- Use a plain clean contrasting background.

Absolute priority:
- Exact same object from the original input image with zero visible changes.
"""
    
    start_time_gen_obj = time.time()
    loop = asyncio.get_running_loop()

    response = await asyncio.wait_for(
        loop.run_in_executor(
            thread_pool,
            lambda: client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[prompt, reference_image],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(image_size="512")
                )
            )
        ),
        timeout = 180    
    )
    end_time_gen_obj = time.time() - start_time_gen_obj

    for candidate in response.candidates:
        for part in candidate.content.parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and inline_data.data:
                with open(output_path, "wb") as f:
                    f.write(inline_data.data)
                return end_time_gen_obj

    raise ValueError(f"No image returned for {obj['object_id']}")
