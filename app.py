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
from collections import defaultdict
from google.oauth2 import service_account

from detect_objects import detect_objects, resize_image
# from detect_objects import detect_objects_fast, resize_image
from generate_object import generate_object_image, safe_name
from generate_metadata import generate_metadata
# from generate_metadata import generate_metadata_fast

# ---------------- INIT ----------------
dir = Path(os.getcwd())
ENV_PATH: Path = dir / '.env'

load_dotenv(ENV_PATH)
# api_key = os.getenv("GEMINI_API_KEY")
# client = genai.Client(api_key=api_key)

def get_client(loc=None):
    try:
        # Vertex JSON auth
        project_id = os.getenv("VERTEX_AI_PROJECT_ID")
        json_path = os.getenv("VERTEX_JSON_PATH")

        if project_id and os.path.exists(json_path):
            credentials = service_account.Credentials.from_service_account_file(
                json_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            return genai.Client(
                vertexai=True,
                project=project_id,
                location= "global" if loc is None else loc,
                credentials=credentials,
                http_options={"timeout": 600000}
            )

        # Nothing configured
        return None

    except Exception as e:
        print(f"Client init failed: {e}")
        return None


client = get_client(loc="global")

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

def clear_selection_state():
    for k in list(st.session_state.keys()):
        if (k.startswith("select_all_") or k.startswith("object_") or 
            k.startswith("meta_select_all_") or k.startswith("m_object_")):
            del st.session_state[k]

# ---- Object Generation Helpers ----
def sync_category(category, objects, parent_key):
    if st.session_state[parent_key]:
        for obj in objects:
            st.session_state[obj["object_id"]] = True
    else:
        for obj in objects:
            st.session_state[obj["object_id"]] = False

def sync_all_categories(category_map, parent_key):
    checked = st.session_state[parent_key]
    for category, objects in category_map.items():
        st.session_state[f"select_all_{category}"] = checked
        for obj in objects:
            st.session_state[obj["object_id"]] = checked

# ---- Metadata Helpers ----
def sync_meta_category(category, objects, parent_key):
    checked = st.session_state.get(parent_key, False)
    for obj in objects:
        st.session_state[f"m_{obj['object_id']}"] = checked

def sync_all_meta_categories(category_map, parent_key):
    checked = st.session_state.get(parent_key, False)
    for category, objects in category_map.items():
        st.session_state[f"meta_select_all_{category}"] = checked
        for obj in objects:
            st.session_state[f"m_{obj['object_id']}"] = checked

# ---------------- ASYNC STREAM GENERATION ----------------
async def run_generation(image, selected_objs, temp_dir, placeholder):
    gen_run_start = time.time()

    clients = [
        (get_client("global"), "global"),
        (get_client("global"), "global"),
        (get_client("global"), "global"),
        (get_client("global"), "global"),
    ]

    clients = [(c, r) for c, r in clients if c is not None]

    semaphores = [asyncio.Semaphore(2) for _ in clients]

    results = []

    async def generate_with_failover(obj, path, start_idx):
        total = len(clients)

        for region_shift in range(total):
            idx = (start_idx + region_shift) % total
            client, region = clients[idx]

            async with semaphores[idx]:
                for attempt in range(2):
                    try:
                        await asyncio.sleep(0.3)

                        t = await generate_object_image(
                            client=client,
                            reference_image=image,
                            obj=obj,
                            output_path=path
                        )

                        return obj, path, t, region

                    except Exception as e:
                        msg = str(e).lower()

                        if "429" in msg or "resource_exhausted" in msg:
                            await asyncio.sleep(1 + attempt)
                            continue
                        break

        raise Exception(f"All regions failed: {obj['object_name']}")

    tasks = []

    for i, obj in enumerate(selected_objs):
        path = os.path.join(
            temp_dir,
            f"{obj['object_id']}_{safe_name(obj['object_name'])}.png"
        )

        tasks.append(
            asyncio.create_task(
                generate_with_failover(obj, path, i % len(clients))
            )
        )

    for completed in asyncio.as_completed(tasks):
        try:
            obj, path, gen_time, region = await completed

            results.append({
                **obj,
                "image_path": path,
                "gen_time": gen_time,
                "region": region,
                "metadata": {}
            })

        except Exception as e:
            st.warning(str(e))

        elapsed = time.time() - gen_run_start

        with placeholder.container():
            st.subheader(
                f"⚡ Live Generation — {len(results)}/{len(selected_objs)} done | ⏱ {elapsed:.1f}s"
            )

            cols_per_row = 5
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
                            st.caption(
                                f"⚡ {r['gen_time']:.2f}s | 🌍 {r.get('region','')}"
                            )

    total_run_time = time.time() - gen_run_start
    total_api_time = sum(r["gen_time"] for r in results)

    return results, total_run_time, total_api_time

# ---------------- ASYNC STREAM METADATA ----------------
async def run_metadata(selected_results, enable_search, placeholder):
    tasks = []
    
    # 1. ADD SEMAPHORE: Limit to 5 concurrent Vertex AI metadata calls
    meta_semaphore = asyncio.Semaphore(5)

    for r in selected_results:
        img = Image.open(r["image_path"])

        async def task_wrapper(res=r, image=img):
            # 2. WRAP WITH SEMAPHORE
            async with meta_semaphore:
                await asyncio.sleep(0.2) # Minor stagger
                m = await generate_metadata(
                    client,
                    image=image,
                    obj=res,
                    search_tool=enable_search
                )
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
            objects, detect_time, search_used_in_detection = asyncio.run(
                # detect_objects_fast(st.session_state.uploaded_image_pil)
                detect_objects(client, st.session_state.uploaded_image_pil)
            )
            app_time = time.time() - start

            clear_selection_state()
            st.session_state.detected_objects = objects
            st.session_state.detect_app_time = app_time

            st.write(f"Detection Time: {detect_time:.2f}s")
            st.write(f"Search Tool used for detection: {search_used_in_detection}")
        
# if "detect_app_time" in st.session_state:
#     st.success(f"Detection App Time: {st.session_state.detect_app_time:.2f}s")
# ---------------- GENERATION ----------------
if st.session_state.detected_objects:
    # st.write(f"Detection Time: {detect_time:.2f}s")
    # st.write(f"Search Tool used for detection: {search_used_in_detection}")
    st.subheader("Select Objects")

    category_map = defaultdict(list)
    for obj in st.session_state.detected_objects:
        category = obj.get("category", "").strip().lower()
        if not category:
            category = obj["object_name"].split()[0].lower()
        category_map[category].append(obj)

    selected = []
    selected_seen = set()


    all_key = "select_all_categories"
    if all_key not in st.session_state:
        st.session_state[all_key] = False

    st.checkbox(
        "Select All Categories",
        key=all_key,
        on_change=sync_all_categories,
        args=(category_map, all_key)
    )

    for category, objects in category_map.items():
        st.markdown(f"### {category.capitalize()}s")

        parent_key = f"select_all_{category}"

        if parent_key not in st.session_state:
            st.session_state[parent_key] = False

        st.checkbox(
            f"Select All {category.capitalize()}s",
            key=parent_key,
            on_change=sync_category,
            args=(category, objects, parent_key)
        )

        cols = st.columns(3)
        for i, obj in enumerate(objects):
            confidence = obj.get("confidence_score", "")
            visibility = obj.get("visibility_score", "")
            visible_desc = obj.get("visible_description", "")

            # if confidence and int(confidence) < 40:
            #     continue
            with cols[i % 3]:
                label = (
                    f"object: {obj['object_name']}  |  "
                    f"location: {obj['position_hint']}  |  "
                    f"visible_description: {visible_desc}  |  "
                    f"confidence: {confidence}  |  "
                    f"visibility: {visibility}"
                )

                if obj["object_id"] not in st.session_state:
                    st.session_state[obj["object_id"]] = False

                if st.checkbox(label, key=obj["object_id"]):
                    key = (
                        obj.get("category", "").strip().lower(),
                        obj.get("object_name", "").strip().lower(),
                        obj.get("position_hint", "").strip().lower()[:40],
                        obj.get("visible_description", "").strip().lower(),
                    )
                    if key not in selected_seen:
                        selected_seen.add(key)
                        selected.append(obj)


    if st.button("Generate"):
        placeholder = st.empty()

        with st.spinner("Generating (streaming)..."):

            results, generation_time, api_time = asyncio.run(
                run_generation(
                    st.session_state.uploaded_image_pil,
                    selected,
                    st.session_state.temp_dir,
                    placeholder
                )
            )
            placeholder.empty()

            avg = api_time / len(results) if results else 0

            st.session_state.generated_results = results
            st.session_state.gen_total = generation_time  
            st.session_state.gen_api_time = api_time  # ✅ sum of all individual times
            st.session_state.gen_avg = avg
            st.session_state.gen_app_time = generation_time
            st.session_state.generation_done = True


    # if "gen_app_time" in st.session_state:
    #     st.success(f"Generation App Time: {st.session_state.gen_app_time:.2f}s")

    if st.session_state.generation_done:
        st.success(f"✅ All done in {st.session_state.gen_total:.2f}s")
        st.write(f"📡 Total API Time (sum): {st.session_state.gen_api_time:.2f}s")
        st.write(f"⚡ Avg per Image: {st.session_state.gen_avg:.2f}s")

        results = st.session_state.generated_results

        cols_per_row = 5
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
                        # ✅ Individual time per image
                        if r.get("gen_time"):
                            st.caption(f"Image {idx + 1} | ⚡ {r['gen_time']:.2f}s")

# ---------------- METADATA ----------------
if st.session_state.generation_done:
    st.subheader("Select for Metadata")

    # Group generated results by category
    meta_category_map = defaultdict(list)
    for r in st.session_state.generated_results:
        category = r.get("category", "").strip().lower()
        if not category:
            category = r["object_name"].split()[0].lower()
        meta_category_map[category].append(r)

    selected_ids = []

    # Global Select All Categories Checkbox
    all_meta_key = "meta_select_all_categories"
    if all_meta_key not in st.session_state:
        st.session_state[all_meta_key] = False

    st.checkbox(
        "Select All Categories",
        key=all_meta_key,
        on_change=sync_all_meta_categories,
        args=(meta_category_map, all_meta_key)
    )

    # Loop through each category
    for category, objects in meta_category_map.items():
        st.markdown(f"### {category.capitalize()}s")

        parent_key = f"meta_select_all_{category}"

        if parent_key not in st.session_state:
            st.session_state[parent_key] = False

        # 1. The Category Checkbox (Only syncs when clicked)
        st.checkbox(
            f"Select All {category.capitalize()}s",
            key=parent_key,
            on_change=sync_meta_category,
            args=(category, objects, parent_key)
        )

        # 2. 3 Columns for Individual Items
        cols = st.columns(3)
        for i, obj in enumerate(objects):
            item_key = f"m_{obj['object_id']}"
            
            # Ensure the state exists
            if item_key not in st.session_state:
                st.session_state[item_key] = False

            with cols[i % 3]:
                label = (
                    f"object: {obj['object_name']}  |  "
                    f"location: {obj['position_hint']}"
                )
                
                # This now allows individual selection!
                if st.checkbox(label, key=item_key):
                    selected_ids.append(obj["object_id"])


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

    if st.session_state.metadata_done:
        st.write(f"Metadata Total: {st.session_state.meta_total:.2f}s")
        st.write(f"Avg Metadata: {st.session_state.meta_avg:.2f}s")

        results = st.session_state.generated_results

        cols_per_row = 4
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
# if detect or gen:
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