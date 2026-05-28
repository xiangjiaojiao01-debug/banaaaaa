from pathlib import Path
import tempfile

from PIL import Image, ImageDraw
import streamlit as st

from backend import CLASS_COLORS, DEFAULT_CONF, MODEL_PATH, predict_and_analyze

st.set_page_config(
    page_title="Banana Detection",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: Arial, sans-serif; }
.block-container { max-width: 1180px; padding-top: 2rem; }
.app-title { font-size: 2rem; font-weight: 700; margin-bottom: .25rem; }
.app-subtitle { color: #666; margin-bottom: 1.5rem; }
.metric-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-top: .75rem; }
.metric-pill { border: 1px solid #ddd; border-radius: 8px; padding: .55rem .75rem; background: #fafafa; }
.det-table { font-size: .9rem; }
</style>
""", unsafe_allow_html=True)


def draw_detections(image: Image.Image, detections):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(label, "red")
        text = f"{label} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_box = draw.textbbox((x1, y1), text)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_h - 5)
        draw.rectangle([x1, text_y, x1 + text_w + 8, text_y + text_h + 6], fill=color)
        draw.text((x1 + 4, text_y + 3), text, fill="black")

    return canvas


def save_upload_to_temp(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def run_uploaded_image(uploaded_file, conf):
    image_path = save_upload_to_temp(uploaded_file)
    image = Image.open(image_path)
    detections, analysis = predict_and_analyze(image_path, conf=conf)
    result_image = draw_detections(image, detections)
    return image, result_image, detections, analysis


st.markdown('<div class="app-title">Banana Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">YOLO backend connected. Only <b>all</b> and <b>s</b> are shown; <b>y</b> is hidden.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.write("Model")
    st.code(str(MODEL_PATH))
    conf = st.slider("Confidence", 0.05, 0.95, float(DEFAULT_CONF), 0.05)

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    tab_upload, tab_camera = st.tabs(["Upload", "Camera"])

    uploaded = None
    source_name = None

    with tab_upload:
        uploaded_file = st.file_uploader("Choose a banana image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_file is not None:
            uploaded = uploaded_file
            source_name = uploaded_file.name

    with tab_camera:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            uploaded = camera_file
            source_name = "camera.jpg"

    if uploaded is not None:
        if st.button("Run detection", type="primary", use_container_width=True):
            with st.spinner("Running YOLO..."):
                original, result, detections, analysis = run_uploaded_image(uploaded, conf)
                st.session_state["last_result"] = {
                    "source_name": source_name,
                    "original": original,
                    "result": result,
                    "detections": detections,
                    "analysis": analysis,
                }

with right:
    if "last_result" not in st.session_state:
        st.info("Upload or take a photo, then run detection.")
    else:
        data = st.session_state["last_result"]
        st.image(data["result"], caption=f"Result: {data['source_name']}", use_container_width=True)

        detections = data["detections"]
        analysis = data["analysis"]
        all_count = sum(1 for d in detections if d["label"] == "all")
        s_count = sum(1 for d in detections if d["label"] == "s")

        st.markdown(
            f"""
            <div class="metric-row">
              <div class="metric-pill">all: <b>{all_count}</b></div>
              <div class="metric-pill">s: <b>{s_count}</b></div>
              <div class="metric-pill">顏色: <b>{analysis['banana_color']}</b></div>
              <div class="metric-pill">黑斑值: <b>{analysis['black_spot_pct']:.2f}%</b></div>
              <div class="metric-pill">黑斑指數: <b>{analysis['black_spot_index']}</b></div>
              <div class="metric-pill">熟度: <b>{analysis['ripeness']}</b></div>
              <div class="metric-pill">甜度: <b>{analysis['sweetness_level']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(analysis["standard"])

        rows = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            rows.append({
                "label": det["label"],
                "confidence": round(det["confidence"], 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)


