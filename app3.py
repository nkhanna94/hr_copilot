import base64
import io
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import openai
from dotenv import load_dotenv

from detect import ProctoringAnalyzer
from scoring import InterviewMonitor

load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def load_image_upload(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv

def extract_frames(video_path, interval_sec=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        return frames
    frame_interval = int(fps * interval_sec)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def generate_llm_bullet_summary(result, card):
    prompt = f"""
You are a professional AI assistant evaluating remote proctoring violations. Summarize the situation in under 3 concise bullet points.

Include:
- Compliance or violation status
- Nature of violation (if any)
- Keep wording short, objective, and clear

Violations: {result.get("violations", "None")}
Score: {card["current_score"]}
Card: {card["card"]}
Reason: {card["reason"]}
Bounding boxes: {result.get("violation_bboxes", [])}

Bullet Summary:
"""

    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for proctoring analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_completion_tokens=300
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("LLM summary error:", e)
        return "- Unable to generate summary"
    
def generate_vision_llm_summary(frame_img_cv):

    img_rgb = cv2.cvtColor(frame_img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    img_b64 = base64.b64encode(img_bytes).decode()

    prompt = """
You are an expert AI proctoring assistant reviewing a candidate's interview video frame for remote proctoring compliance. Carefully analyze the image and provide a concise summary (2-3 bullet points) covering:

- Whether the candidate is clearly visible and focused
- Any suspicious behaviors (e.g., multiple people, unusual objects, cheating attempts)
- Issues that could affect interview validity (e.g., poor lighting, face obstruction)
- Conclude if the frame shows compliance or violation, and briefly explain why.

Keep the summary objective, clear, and professional.

"""

    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for proctoring analysis."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": f"<image>{img_b64}</image>"}  
            ],
            temperature=1.0,
            max_completion_tokens=300
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("Vision LLM summary error:", e)
        return "- Unable to generate vision-based summary"


def analyze_proctoring_session(ref_img_cv, test_imgs_cv, analyzer, monitor, temp_dir="custom_temp"):
    os.makedirs(temp_dir, exist_ok=True)
    scores = []
    summaries = []

    for i, test_img_cv in enumerate(test_imgs_cv):
        with tempfile.NamedTemporaryFile(suffix=".png", dir=temp_dir, delete=False) as ref_tmp, \
             tempfile.NamedTemporaryFile(suffix=".png", dir=temp_dir, delete=False) as test_tmp:
            cv2.imwrite(ref_tmp.name, ref_img_cv)
            cv2.imwrite(test_tmp.name, test_img_cv)

            result = analyzer.analyze_dual(ref_tmp.name, test_tmp.name)
            card = monitor.evaluate_card(result)

            llm_caption = generate_llm_bullet_summary(result, card)
            vision_llm_caption = generate_vision_llm_summary(test_img_cv)

            summaries.append({
                "index": i,
                "score": card['current_score'],
                "card": card['card'],
                "caption_by_llm": llm_caption,
                "caption_by_vision_llm": vision_llm_caption,
                "violations": result.get("violations", ""),
                "violation_bboxes": result.get("violation_bboxes", [])
            })
            
        os.remove(ref_tmp.name)
        os.remove(test_tmp.name)
        scores.append(card["current_score"])

        if card['card'] == "Red 🔴" or card['current_score'] <= 40:
            break

    return scores, summaries

# --- Streamlit UI ---
st.set_page_config(page_title="Proctoring Analyzer", layout="wide")
st.title("Proctoring Analyzer")
st.write("Upload a reference image and test image or a video for automated proctoring evaluation.")

if "analyzer" not in st.session_state:
    st.session_state.analyzer = ProctoringAnalyzer()
if "monitor" not in st.session_state:
    st.session_state.monitor = InterviewMonitor()

analyzer = st.session_state.analyzer
monitor = st.session_state.monitor

col1, col2 = st.columns(2)

with col1:
    ref_img_file = st.file_uploader("Reference Image (ID/Selfie)", type=["jpg", "jpeg", "png"], key="ref")
    if ref_img_file:
        st.image(ref_img_file, caption="Reference Image", use_container_width=True)

with col2:
    test_img_file = st.file_uploader("Test Image (Live Webcam)", type=["jpg", "jpeg", "png"], key="test")
    if test_img_file:
        st.image(test_img_file, caption="Test Image", use_container_width=True)

video_file = st.file_uploader("Or Upload a Video (MP4, AVI)", type=["mp4", "avi"], key="video")

custom_temp_dir = os.path.join(os.getcwd(), "custom_temp")
os.makedirs(custom_temp_dir, exist_ok=True)

if video_file:
    st.info("Processing video. Extracting frames...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(video_file.read())
        video_path = tmp_vid.name

    frames = extract_frames(video_path)
    if not frames:
        st.error("Failed to extract frames.")
    else:
        ref_frame = frames[0]
        test_frames = frames[1:]
        st.image(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB), caption="Reference Frame")

        scores, summaries = analyze_proctoring_session(ref_frame, test_frames, analyzer, monitor, temp_dir=custom_temp_dir)

        for summary in summaries:
            col_detail, col_frame = st.columns([1, 1])

            with col_detail:
                st.markdown(f"### Frame {summary['index']+1}")
                st.markdown("**Text-based LLM Summary:**")
                st.markdown(summary['caption_by_llm'])

                st.markdown("**Vision-enabled LLM Summary:**")
                st.markdown(summary['caption_by_vision_llm'])

                if summary["card"] == "Red 🔴":
                    st.error("Red Card 🚫")
                elif summary["card"] == "Amber 🟡":
                    st.warning("Amber Card ⚠️")
                else:
                    st.success("Green Card ✅")

            with col_frame:
                frame_img = frames[summary['index']+1].copy()
                color_map = {'g': (137, 180, 62), 'a': (0, 191, 255), 'r': (60, 20, 220)}
                bbox_color = color_map.get(summary['card'][0].lower(), (255, 255, 255))

                for item in summary["violation_bboxes"]:
                    bbox = item.get("bbox", [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame_img, (x1, y1), (x2, y2), bbox_color, 2)
                        cls_name = item.get("class", "object")
                        cv2.putText(frame_img, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

                frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                st.image(frame_img_rgb, caption=f"Frame {summary['index']+1}", use_container_width=True)

            st.markdown("---")

        final_score = scores[-1] if scores else 100
        final_card = summaries[-1]['card'] if summaries else "Green 🟢"

        st.markdown("## Final Verdict")
    
        table_data = []

        for s in summaries:
            if s.get("violations"):
                table_data.append({
                    "Frame": s["index"] + 1,
                    "Violation": s["violations"],
                    "Score": s["score"]
                })

        if table_data:
            st.markdown("#### Violations Overview")
            st.table(table_data)


        st.write(f"**Final Score:** {final_score}")
        if final_card == "Red 🔴":
            st.error("🚫 Interview Terminated due to critical violations.")
        elif final_card == "Amber 🟡":
            st.warning("⚠️ Minor issues detected. Continue monitoring.")
        else:
            st.success("✅ Interview conditions ideal throughout the session.")

    os.remove(video_path)

elif test_img_file:
    ref_img_cv = load_image_upload(ref_img_file)
    test_img_cv = load_image_upload(test_img_file)

    scores, summaries = analyze_proctoring_session(ref_img_cv, [test_img_cv], analyzer, monitor, temp_dir=custom_temp_dir)
    summary = summaries[0]

    st.markdown("## Proctoring Result")
    st.markdown(summary['caption_by_llm'])

    if summary["card"] == "Red 🔴":
        st.error("🚫 Interview Terminated due to critical violations.")
    elif summary["card"] == "Amber 🟡":
        st.warning("⚠️ Minor issues detected. Continue monitoring.")
    else:
        st.success("✅ Interview conditions ideal. No violations detected.")

else:
    st.info("Upload reference & test image or a video to begin analysis.")

st.caption("Close the browser tab to end the session.")