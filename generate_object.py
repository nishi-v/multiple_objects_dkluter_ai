import os
import re
from typing import Dict
from PIL import Image
from google.genai import types
from google.genai.client import Client
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

thread_pool = ThreadPoolExecutor(max_workers=3)

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")

async def generate_object_image(
    client: Client,
    reference_image: Image.Image,
    obj: Dict[str, str],
    output_path: str
) ->float:
    prompt = f"""
You are an expert single-object completion model.

Task:
Generate ONLY the selected object from the input image.

Input:
- object_name: {obj['object_name']}
- category: {obj.get('category', obj['object_name'])}
- position_hint: {obj.get('position_hint', '')}
- short_description: {obj.get('short_description', '')}
- visible_description: {obj.get('visible_description', '')}

PRIMARY RULE:
- First visually locate the exact selected object inside the input image.
- Use the real visible object in the image as the MAIN reference.
- Then complete only the missing / hidden / cropped parts of that SAME object.
- Do NOT generate a new version.
- Do NOT replace with a similar object.

OBJECT MATCH RULE:
Use all provided data to identify the correct object:
- object_name
- category
- position_hint
- short_description
- visible_description

If multiple similar objects exist:
- choose the one matching position + color + shape + design + text clues.

REFERENCE PRIORITY:
1. Input image visible object (highest priority)
2. visible_description
3. short_description
4. object_name / category
5. normal real-world knowledge for hidden parts only

STRICT PRESERVATION RULE:
The visible portion must remain true to the original object.

Do NOT change visible:
- color
- material
- texture
- graphics
- logo
- text
- typography
- pattern
- design
- shape
- wear marks
- folds
- proportions

Do NOT:
- redesign
- beautify
- sharpen
- recolor
- modernize
- replace branding
- invent new text

COMPLETION RULE:
- Extend naturally from visible boundaries.
- Generate only unseen / missing areas.
- Keep same object identity.
- Keep same construction style.
- Keep realistic proportions.
- Keep same age / wear level.

DETAIL RULE:
Use provided data to help complete accurately:
- visible_description helps with colors, materials, text, design.
- short_description helps with object type/use.
- object_name helps identity.
- category helps structure.

BOOK / PRINTED OBJECT RULE:
If selected object is book / notebook / magazine / box:
- preserve visible cover design exactly in spirit.
- preserve title / author text if visible.
- complete hidden spine/back consistently.
- do NOT invent a different edition.

ANTI-HALLUCINATION:
- No extra accessories.
- No extra objects.
- No mixed nearby object features.
- No fake brands.
- No random added text.

BACKGROUND RULE:
- Plain clean neutral background.
- One object only.
- Full object visible.

OUTPUT:
- Same original selected object
- Completed realistically
- Visible region preserved
- Missing parts completed
- Plain background
"""

    start_time_gen_obj = time.time()
    loop = asyncio.get_running_loop()

    response = await asyncio.wait_for(
        loop.run_in_executor(
            thread_pool,
            lambda: client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                # model="gemini-2.5-flash-image",
                contents=[prompt, reference_image],
                config=types.GenerateContentConfig(
                    temperature=0.02,
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(image_size="512", aspect_ratio="1:1"),
                    # image_config=types.ImageConfig(aspect_ratio="1:1"),
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
