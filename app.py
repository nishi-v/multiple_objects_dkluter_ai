import os
import json
import math
import tempfile
import zipfile
from io import BytesIO
import streamlit as st
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import asyncio
import time

from detect_objects import detect_objects, resize_image
# from detect_objects import detect_objects_fast, resize_image
from generate_object import generate_object_image, safe_name
from generate_metadata import generate_metadata
# from generate_metadata import generate_metadata_fast

# ---------------- INIT ----------------
dir = Path(os.getcwd())
ENV_PATH: Path = dir / '.env'

load_dotenv(ENV_PATH)
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

st.set_page_config(page_title="D'kluter AI Studio", layout="wide")

# ---------------- SESSION ----------------
def ensure_session_state():
    defaults = {
        "uploaded_image_pil": None,
        "detected_objects": None,
        "generated_results": [],
        "temp_dir": None,
        "generation_done": False,
        "metadata_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_session_state()

# ---------------- ASYNC STREAM GENERATION ----------------
async def run_generation(image, selected_objs, temp_dir, placeholder):
    tasks = []

    for obj in selected_objs:
        path = os.path.join(
            temp_dir,
            f"{obj['object_id']}_{safe_name(obj['object_name'])}.png"
        )

        async def task_wrapper(o=obj, p=path):
            t = await generate_object_image(client, image, o, p)
            return o, p, t  # ✅ bundle everything

        tasks.append(asyncio.create_task(task_wrapper()))

    results = []
    total_time = 0

    for completed in asyncio.as_completed(tasks):
        obj, path, gen_time = await completed

        total_time += gen_time

        result = {
            **obj,
            "image_path": path,
            "gen_time": gen_time,
            "metadata": {}
        }

        results.append(result)

        # 🔥 STREAM UI
        with placeholder.container():
            st.subheader("⚡ Live Generation")
            for r in results:
                st.image(r["image_path"], width=150)
                st.write(f"{r['object_name']} → {r['gen_time']:.2f}s")

    return results, total_time

# ---------------- ASYNC STREAM METADATA ----------------
async def run_metadata(selected_results, enable_search, placeholder):
    tasks = []

    for r in selected_results:
        img = Image.open(r["image_path"])

        async def task_wrapper(res=r, image=img):
            m = await generate_metadata(
                client,
                image=image,
                obj=res,
                search_tool=enable_search
            )
            # m = await generate_metadata_fast(
            #     image=image,
            #     obj=res,
            # )
            return res["object_id"], m

        tasks.append(asyncio.create_task(task_wrapper()))

    results_map = {}

    for completed in asyncio.as_completed(tasks):
        obj_id, meta = await completed

        results_map[obj_id] = meta

        # 🔥 STREAM UI
        with placeholder.container():
            st.subheader("⚡ Live Metadata")
            for oid, m in results_map.items():
                st.write(
                    f"{oid} → {m['Time Taken']:.2f}s | "
                    f"Tokens: {m['Output Token Count']} | "
                    f"Search: {m['Search Tool Used']}"
                )

    return results_map

# ---------------- UI ----------------
st.title("D’kluter Multi-Object AI Studio")

enable_search = st.sidebar.checkbox("Enable Search", True)

uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp()

    path = os.path.join(st.session_state.temp_dir, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    img = Image.open(path)
    st.image(img, width=300)

    img_resized = resize_image(img)

    st.session_state.uploaded_image_pil = img_resized

# ---------------- DETECTION ----------------
if st.session_state.uploaded_image_pil is not None:
    if st.button("Detect Objects"):
        with st.spinner("Detecting..."):
            start = time.time()
            objects, detect_time = asyncio.run(
                # detect_objects_fast(st.session_state.uploaded_image_pil)
                detect_objects(client, st.session_state.uploaded_image_pil)
            )
            app_time = time.time() - start

            st.session_state.detected_objects = objects
            st.session_state.detect_app_time = app_time

            st.write(f"Detection Time: {detect_time:.2f}s")
        
# if "detect_app_time" in st.session_state:
#     st.success(f"Detection App Time: {st.session_state.detect_app_time:.2f}s")
# ---------------- GENERATION ----------------
if st.session_state.detected_objects:
    st.subheader("Select Objects")

    selected = []
    for obj in st.session_state.detected_objects:
        label = (
            f"object: {obj['object_name']}  |  "
            f"location: {obj['position_hint']}"
        )

        if st.checkbox(label, key=obj["object_id"]):
        # if st.checkbox(obj["object_name"], key=obj["object_id"]):
            selected.append(obj)

    if st.button("Generate"):
        placeholder = st.empty()

        with st.spinner("Generating (streaming)..."):
            start = time.time()

            results, total = asyncio.run(
                run_generation(
                    st.session_state.uploaded_image_pil,
                    selected,
                    st.session_state.temp_dir,
                    placeholder
                )
            )
            app_time = time.time() - start
            placeholder.empty()

            avg = total / len(results)

            st.session_state.generated_results = results
            st.session_state.gen_total = total
            st.session_state.gen_avg = avg
            st.session_state.gen_app_time = app_time
            st.session_state.generation_done = True

    # if "gen_app_time" in st.session_state:
    #     st.success(f"Generation App Time: {st.session_state.gen_app_time:.2f}s")

    if st.session_state.generation_done:
        st.write(f"Total Gen Time: {st.session_state.gen_total:.2f}s")
        st.write(f"Avg Time/Image: {st.session_state.gen_avg:.2f}s")

        # 🔥 PERSISTENT GRID (THIS FIXES DISAPPEAR ISSUE)
        results = st.session_state.generated_results

        cols_per_row = 3
        rows = math.ceil(len(results) / cols_per_row)

        for i in range(rows):
            cols = st.columns(cols_per_row)

            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx >= len(results):
                    continue

                r = results[idx]

                with cols[j]:
                    with st.container(border=True):
                        st.image(r["image_path"], use_container_width=True)
                        st.success(r["object_name"])

                        if r.get("gen_time"):
                            st.write(f"⚡ {r['gen_time']:.2f}s")

# ---------------- METADATA ----------------
if st.session_state.generation_done:
    st.subheader("Select for Metadata")

    selected_ids = []
    for r in st.session_state.generated_results:
        label = (
            f"object: {r['object_name']}  |  "
            f"location: {r['position_hint']}"
        )

        if st.checkbox(label, key=f"m_{r['object_id']}"):
            selected_ids.append(r["object_id"])

    if st.button("Generate Metadata"):
        placeholder = st.empty()

        selected_results = [
            r for r in st.session_state.generated_results
            if r["object_id"] in selected_ids
        ]

        with st.spinner("Metadata streaming..."):
            start = time.time()

            results_map = asyncio.run(
                run_metadata(
                    selected_results,
                    enable_search,
                    placeholder
                )
            )

            app_time = time.time() - start
            meta_times = [
                m["Time Taken"]
                for m in results_map.values()
                if m and "Time Taken" in m
            ]

            total_meta_time = sum(meta_times)
            avg_meta_time = total_meta_time / len(meta_times) if meta_times else 0

            updated = []
            for r in st.session_state.generated_results:
                if r["object_id"] in results_map:
                    r["metadata"] = results_map[r["object_id"]]
                updated.append(r)

            st.session_state.generated_results = updated
            st.session_state.meta_total = total_meta_time
            st.session_state.meta_avg = avg_meta_time
            st.session_state.meta_app_time = app_time
            st.session_state.metadata_done = True

    # if "meta_app_time" in st.session_state:
    #     st.success(f"Metadata App Time: {st.session_state.meta_app_time:.2f}s")

    if st.session_state.metadata_done:
        st.write(f"Metadata Total: {st.session_state.meta_total:.2f}s")
        st.write(f"Avg Metadata: {st.session_state.meta_avg:.2f}s")

        results = st.session_state.generated_results

        cols_per_row = 2
        total = len(results)
        rows = math.ceil(total / cols_per_row)

        for i in range(rows):
            cols = st.columns(cols_per_row)

            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx >= total:
                    continue

                r = results[idx]

                if not r.get("metadata"):
                    continue

                m = r["metadata"]

                with cols[j]:
                    with st.container(border=True):

                        # 🔥 HEADER
                        st.markdown(f"### {r['object_name']}")

                        # 🔥 IMAGE
                        st.image(r["image_path"], use_container_width=True)

                        # 🔥 GEN TIME
                        # if r.get("gen_time") is not None:
                        #     st.success(f"⚡ Gen Time: {r['gen_time']:.2f}s")

                        # 🔥 META STATS
                        c1, c2 = st.columns(2)

                        with c1:
                            st.info(f"⏱ Meta: {m['Time Taken']:.2f}s")
                            st.info(f"📥 Input: {m['Input Token Count']}")

                        with c2:
                            st.info(f"📤 Output: {m['Output Token Count']}")
                            st.info(f"🔍 Search: {m['Search Tool Used']}")

                        # 🔥 TITLE + DESC
                        st.markdown("#### 🧾 Details")
                        st.success(f"**Title:** {m['Title']}")
                        st.info(f"**Desc:** {m['Description']}")

                        # 🔥 TAGS (BOXED)
                        if m.get("Tags"):
                            st.markdown("#### 🏷 Tags")
                            for tag in m["Tags"]:
                                with st.container(border=True):
                                    for k, v in tag.items():
                                        st.write(f"**{k}:** {v}")

                        # 🔥 FIELDS (BOXED)
                        if m.get("Fields"):
                            st.markdown("#### 📦 Fields")
                            for field in m["Fields"]:
                                with st.container(border=True):
                                    for k, v in field.items():
                                        st.write(f"**{k}:** {v}")

# ---------------- APP TIMINGS ----------------
detect = st.session_state.get("detect_app_time", 0)
gen = st.session_state.get("gen_app_time", 0)
meta = st.session_state.get("meta_app_time", 0)

if detect or gen or meta:
    st.markdown("## ⏱ App Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success(f"Detection: {detect:.2f}s")

    with col2:
        st.success(f"Generation: {gen:.2f}s")

    with col3:
        st.success(f"Metadata: {meta:.2f}s")

    total = detect + gen + meta
    st.info(f"🚀 Total App Time: {total:.2f}s")

# ---------------- DOWNLOAD ----------------
if st.session_state.generated_results:
    results = st.session_state.generated_results

    buf = BytesIO()
    final_metadata = []

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:

        # 🔥 ADD IMAGES
        for r in results:
            if r.get("image_path") and os.path.exists(r["image_path"]):
                filename = os.path.basename(r["image_path"])
                z.write(r["image_path"], f"images/{filename}")

            # 🔥 ADD METADATA
            if r.get("metadata"):
                m = r["metadata"]

                final_metadata.append({
                    "object_id": r.get("object_id"),
                    "object_name": r.get("object_name"),
                    "position_hint": r.get("position_hint"),
                    "image_name": os.path.basename(r.get("image_path", "")),

                    "title": m.get("Title"),
                    "description": m.get("Description"),
                    "tags": m.get("Tags", []),
                    "fields": m.get("Fields", []),

                    "gen_time": r.get("gen_time"),
                    "metadata_time": m.get("Time Taken"),
                    "input_tokens": m.get("Input Token Count"),
                    "output_tokens": m.get("Output Token Count"),
                    "search_used": m.get("Search Tool Used"),
                })

        # 🔥 ADD JSON
        z.writestr(
            "metadata.json",
            json.dumps(final_metadata, indent=2, ensure_ascii=False)
        )

    buf.seek(0)

    # ✅ SINGLE CLICK DOWNLOAD
    st.download_button(
        label="⬇️ Download Results (Images + Metadata)",
        data=buf,
        file_name="results.zip",
        mime="application/zip"
    )