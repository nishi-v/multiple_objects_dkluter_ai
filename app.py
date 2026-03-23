import os
import json
import math
import tempfile
import zipfile
from io import BytesIO
import streamlit as st
from PIL import Image

from detect_objects import init_client as init_detect_client, detect_objects
from generate_object import init_client as init_generate_client, generate_object_image, safe_name
from generate_metadata import generate_metadata

st.set_page_config(page_title="Multiple Object Detection - D'kluter", layout="wide")

def ensure_session_state():
    defaults = {
        "uploaded_image_path": None,
        "uploaded_image_pil": None,
        "detected_objects": None,
        "generated_results": [],
        "temp_dir": None,
        "last_uploaded_name": None,
        "generation_done": False,
        "metadata_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_state():
    for key in [
        "uploaded_image_path",
        "uploaded_image_pil",
        "detected_objects",
        "generated_results",
        "temp_dir",
        "last_uploaded_name",
        "generation_done",
        "metadata_done",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    ensure_session_state()

def save_uploaded_file(uploaded_file: str, temp_dir: str) -> str:
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def display_tags(tags):
    st.markdown("**Tags:**")
    if not tags:
        st.write("No tags")
        return
    for i, tag in enumerate(tags, 1):
        st.markdown(f"**Tag {i}:**")
        if isinstance(tag, dict):
            st.code("\n".join([f"{k}: {v}" for k, v in tag.items()]))
        else:
            st.code(str(tag))

def display_fields(fields):
    st.markdown("**Fields:**")
    if not fields:
        st.write("No fields")
        return
    for i, field in enumerate(fields, 1):
        st.markdown(f"**Field {i}:**")
        if isinstance(field, dict):
            st.code("\n".join([f"{k}: {v}" for k, v in field.items()]))
        else:
            st.code(str(field))

def display_generated_card(result):
    metadata = result.get("metadata", {})

    st.markdown(f"### {result.get('object_id', '')} — {result.get('object_name', '')}")
    img_col, info_col, field_col = st.columns([1, 2, 2])

    with img_col:
        image_path = result.get("image_path")
        if image_path and os.path.exists(image_path):
            st.image(image_path, width=200)

    with info_col:
        st.write(f"**Name:** {result.get('object_name', '')}")
        st.write(f"**Position:** {result.get('position_hint', '')}")

        if metadata:
            st.write(f"**Title:** {metadata.get('Title', '')}")
            st.write(f"**Description:** {metadata.get('Description', '')}")
            display_tags(metadata.get("Tags", []))

    with field_col:
        if metadata:
            display_fields(metadata.get("Fields", []))

    st.markdown("---")

def render_generated_grid(results):
    cols_per_row = 2
    total = len(results)
    rows = math.ceil(total / cols_per_row)

    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            item_idx = row_idx * cols_per_row + col_idx
            if item_idx >= total:
                continue

            result = results[item_idx]
            with cols[col_idx]:
                st.markdown(f"**{result['object_name']}**")
                st.caption(f"Position: {result['position_hint']}")

                if result.get("image_path") and os.path.exists(result["image_path"]):
                    st.image(result["image_path"], use_container_width=True)

def create_download_zip(results, original_image_path):
    zip_buffer = BytesIO()
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    final_metadata = []
    used_names = set()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, res in enumerate(results, 1):
            image_path = res.get("image_path")
            obj_name = res.get("object_name", f"object_{idx}")
            safe_obj_name = safe_name(obj_name)
            stub = f"{base_name}_{safe_obj_name}"

            if stub in used_names:
                stub = f"{stub}_{idx}"
            used_names.add(stub)

            ext = os.path.splitext(image_path)[1] if image_path else ".png"
            ext = ext or ".png"
            image_name = f"{stub}{ext}"

            if image_path and os.path.exists(image_path):
                zf.write(image_path, f"images/{image_name}")

            if res.get("metadata"):
                metadata = res["metadata"]
                final_metadata.append({
                    "object_id": res.get("object_id"),
                    "object_name": res.get("object_name"),
                    "position_hint": res.get("position_hint"),
                    "image_name": image_name,
                    "title": metadata.get("Title", ""),
                    "description": metadata.get("Description", ""),
                    "tags": metadata.get("Tags", []),
                    "fields": metadata.get("Fields", [])
                })

        metadata_filename = f"{base_name}_metadata.json"
        zf.writestr(metadata_filename, json.dumps(final_metadata, indent=2, ensure_ascii=False))

    zip_buffer.seek(0)
    return zip_buffer

st.title("Object Selection, Generation, and Metadata")
ensure_session_state()

with st.sidebar:
    st.header("Settings")
    enable_search_tool = st.checkbox("Enable Google Search Tool", value=False)

    if st.button("Reset"):
        reset_state()
        st.rerun()

uploaded_file = st.file_uploader(
    "Upload one input image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp()

    if st.session_state.last_uploaded_name != uploaded_file.name:
        saved_path = save_uploaded_file(uploaded_file, st.session_state.temp_dir)
        st.session_state.uploaded_image_path = saved_path
        st.session_state.uploaded_image_pil = Image.open(saved_path)
        st.session_state.detected_objects = None
        st.session_state.generated_results = []
        st.session_state.generation_done = False
        st.session_state.metadata_done = False
        st.session_state.last_uploaded_name = uploaded_file.name

    st.image(st.session_state.uploaded_image_pil, caption="Input Image", width=320)

    if st.session_state.detected_objects is None:
        if st.button("Detect Objects", type="primary"):
            with st.spinner("Detecting objects..."):
                client = init_detect_client()
                st.session_state.detected_objects = detect_objects(
                    client,
                    st.session_state.uploaded_image_pil
                )
                st.session_state.generated_results = []
                st.session_state.generation_done = False
                st.session_state.metadata_done = False
            st.rerun()

if st.session_state.detected_objects is not None:
    detected_objects = st.session_state.detected_objects

    if not detected_objects:
        st.warning("No objects detected.")
    else:
        st.subheader("1. Select Objects to Generate")

        selected_for_generation = []
        for obj in detected_objects:
            checked = st.checkbox(
                f"{obj['object_name']} — {obj['position_hint']}",
                key=f"gen_select_{obj['object_id']}"
            )
            if checked:
                selected_for_generation.append(obj)

        if st.button("Generate Selected Objects", type="primary"):
            if not selected_for_generation:
                st.warning("Select at least one object.")
            else:
                with st.spinner("Generating selected object images..."):
                    gen_client = init_generate_client()
                    generated_results = []

                    for obj in selected_for_generation:
                        generated_path = os.path.join(
                            st.session_state.temp_dir,
                            f"{obj['object_id']}_{safe_name(obj['object_name'])}_generated.png"
                        )

                        generate_object_image(
                            client=gen_client,
                            reference_image=st.session_state.uploaded_image_pil,
                            obj=obj,
                            output_path=generated_path
                        )

                        generated_results.append({
                            "object_id": obj["object_id"],
                            "object_name": obj["object_name"],
                            "short_description": obj["short_description"],
                            "position_hint": obj["position_hint"],
                            "status": "generated",
                            "image_path": generated_path,
                            "metadata": {}
                        })

                    st.session_state.generated_results = generated_results
                    st.session_state.generation_done = True
                    st.session_state.metadata_done = False
                st.rerun()

if st.session_state.generation_done and st.session_state.generated_results:
    st.subheader("2. Generated Images")
    render_generated_grid(st.session_state.generated_results)

    st.subheader("3. Select Generated Images for Metadata")

    selected_for_metadata_ids = []
    for result in st.session_state.generated_results:
        checked = st.checkbox(
            f"{result['object_name']} — {result['position_hint']}",
            key=f"meta_select_{result['object_id']}"
        )
        if checked:
            selected_for_metadata_ids.append(result["object_id"])

    if st.button("Generate Metadata for Selected Images", type="primary"):
        if not selected_for_metadata_ids:
            st.warning("Select at least one generated image.")
        else:
            with st.spinner("Generating metadata..."):
                updated_results = []

                for result in st.session_state.generated_results:
                    if result["object_id"] in selected_for_metadata_ids:
                        generated_img = Image.open(result["image_path"])
                        metadata_result = generate_metadata(
                            image=generated_img,
                            obj=result,
                            search_tool=enable_search_tool
                        )
                        updated = dict(result)
                        updated["metadata"] = metadata_result
                        updated_results.append(updated)
                    else:
                        updated_results.append(result)

                st.session_state.generated_results = updated_results
                st.session_state.metadata_done = True
            st.rerun()

if st.session_state.metadata_done and st.session_state.generated_results:
    metadata_results = [r for r in st.session_state.generated_results if r.get("metadata")]

    if metadata_results:
        st.subheader("4. Final Results")
        for result in metadata_results:
            display_generated_card(result)

if st.session_state.generated_results and st.session_state.uploaded_image_path:
    zip_file = create_download_zip(
        st.session_state.generated_results,
        st.session_state.uploaded_image_path
    )
    base_name = os.path.splitext(os.path.basename(st.session_state.uploaded_image_path))[0]

    st.download_button(
        "Download Images + Metadata",
        data=zip_file,
        file_name=f"{base_name}_results.zip",
        mime="application/zip"
    )