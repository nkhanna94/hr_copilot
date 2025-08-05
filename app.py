import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

from detect import ProctoringAnalyzer
from scoring import InterviewMonitor

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

        os.remove(ref_tmp.name)
        os.remove(test_tmp.name)

        card = monitor.evaluate_card(result)
        scores.append(card['current_score'])

        summaries.append({
            "index": i,
            "score": card['current_score'],
            "card": card['card'],
            "reason": card['reason'],
            "caption": result.get("caption", ""),
            "violations": result.get("caption", "").split("VIOLATIONS:")[-1].strip() if "VIOLATIONS:" in result.get("caption", "") else None
        })

        if card['card'] == "Red":
            break  

    return scores, summaries

st.set_page_config(page_title="Proctoring Analyzer", layout="wide")
st.title("Proctoring Analyzer")
st.write("Upload either a reference image and test image, or a video for automated analysis.")

if "analyzer" not in st.session_state:
    st.session_state.analyzer = ProctoringAnalyzer()
if "monitor" not in st.session_state:
    st.session_state.monitor = InterviewMonitor()

analyzer = st.session_state.analyzer
monitor = st.session_state.monitor

col1, col2 = st.columns(2)

with col1:
    ref_img_file = st.file_uploader("Reference image (ID/selfie)", type=["jpg", "jpeg", "png"], key="ref")
    if ref_img_file:
        st.image(ref_img_file, caption="Reference image", use_container_width=True)

with col2:
    test_img_file = st.file_uploader("Current webcam/image", type=["jpg", "jpeg", "png"], key="test")
    if test_img_file:
        st.image(test_img_file, caption="Test image", use_container_width=True)

video_file = st.file_uploader("Or upload video (MP4, AVI)", type=["mp4", "avi"], key="video")

custom_temp_dir = os.path.join(os.getcwd(), "custom_temp")
os.makedirs(custom_temp_dir, exist_ok=True)

if video_file:
    st.info("Processing video. Extracting frames every 1 second...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(video_file.read())
        video_path = tmp_vid.name

    frames = extract_frames(video_path, interval_sec=1)

    if not frames:
        st.error("Failed to extract frames from video.")
    else:
        st.success(f"Extracted {len(frames)} frames from video.")
        ref_frame = frames[0]
        test_frames = frames[1:]

        st.image(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB), caption="Reference Frame (First Frame)")

        scores, summaries = analyze_proctoring_session(ref_frame, test_frames, analyzer, monitor, temp_dir=custom_temp_dir)

        # for summary in summaries:
        #     st.markdown(f"### Frame {summary['index']+1}")
        #     st.write(f"Score: {summary['score']}, Card: {summary['card']}")
        #     st.write(summary["caption"])
        #     if summary["violations"]:
        #         st.warning(summary["violations"])
        #     st.markdown("---")

        #     if summary['card'] == "Red":
        #         st.error("🚫 Interview Terminated due to critical violations.")
        #         break

        for summary in summaries:
            col_detail, col_frame = st.columns([1, 1])  # equal width columns

            with col_detail:
                st.markdown(f"### Frame {summary['index']+1} Details")
                st.write(f"**Score:** {summary['score']}")
                st.write(f"**Card:** {summary['card']}")
                st.write(f"**Reason:** {summary['reason']}")
                st.write(summary["caption"])
                if summary["violations"]:
                    st.warning(summary["violations"])

            with col_frame:
                frame_img_rgb = cv2.cvtColor(frames[summary['index']+1], cv2.COLOR_BGR2RGB)
                st.image(frame_img_rgb, caption=f"Frame {summary['index']+1}", use_container_width=True)

            st.markdown("---")


        final_score = scores[-1] if scores else 100
        final_card = "Green"
        if any(s['card'] == "Red" for s in summaries):
            final_card = "Red"
        elif any(s['card'] == "Amber" for s in summaries):
            final_card = "Amber"

        st.markdown("## Video Summary")
        st.write(f"Final Score: {final_score:.1f}")
        st.write(f"Final Card: {final_card}")

        if final_score <= 40 or final_card == "Red":
            st.error("🚫 Interview Terminated due to critical violations in video.")
        elif final_card == "Amber":
            st.warning("⚠️ Minor issues detected during video. Continue monitoring.")
        else:
            st.success("✅ Interview conditions ideal throughout the video.")

    os.remove(video_path)

elif ref_img_file and test_img_file:
    ref_img_cv = load_image_upload(ref_img_file)
    test_img_cv = load_image_upload(test_img_file)

    scores, summaries = analyze_proctoring_session(ref_img_cv, [test_img_cv], analyzer, monitor, temp_dir=custom_temp_dir)

    summary = summaries[0]
    st.markdown("## Proctoring Result")
    st.write(f"**Score:** {summary['score']}")
    st.write(f"**Card:** {summary['card']}")
    st.write(f"**Reason:** {summary['reason']}")
    if summary["violations"]:
        st.warning(summary["violations"])
    else:
        st.success(summary["caption"])

    if summary["card"] == "Red" or summary["score"] <= 40:
        st.error("🚫 Interview Terminated due to critical violations.")
    elif summary["card"] == "Amber":
        st.warning("⚠️ Minor issues detected. Continue monitoring.")
    else:
        st.success("✅ Interview conditions ideal. No violations detected.")

else:
    st.info("Please upload either a pair of images or a video to analyze.")

st.caption("Close the browser tab to end the session.")