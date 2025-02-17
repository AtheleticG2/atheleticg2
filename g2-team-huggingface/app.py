import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import tempfile
import pandas as pd
import subprocess
import os
from sprintstart_criteria_checks import evaluate_sprint_start, get_player_coords
from sprintrunning_criteria_checks import evaluate_sprint_running
from longjump_criteria_checks import evaluate_long_jump
from highjump_criteria_checks import evaluate_high_jump
from shotput_criteria_checks import evaluate_shot_put
from discusthrow_criteria_check import evaluate_discus_throw
from javelin_criteria_checks import evaluate_javelin_throw
from hurdling_criteria_checks import evaluate_hurdling

# Set page config
st.set_page_config("Athlete Assist", layout="wide")

st.title("Athlete Assist")
coach_image_2 = "coach_2.png"  

st.sidebar.image(coach_image_2)

# hugging-face markdown
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@100..900&display=swap" rel="stylesheet">
    <style>
        body, h1, h2, h3, h4, h5, h6, div, span, p, a, li, ul {
            font-family: 'Geist', sans-serif !important;
        }
        .streamlit-table th, .streamlit-table td {
            white-space: normal;
        }
    </style>
""", unsafe_allow_html=True)

# # Sport Selection
sport = st.sidebar.selectbox(
    "**Which sport would you like to analyze?**",
    ("Sprint Starting Technique", "Sprint Running Technique", "Long Jump", "High Jump", 'Discus Throw','Javelin Throw','Shotput'),
)

st.write(f"**You selected:** {sport}")
uploaded_file = st.sidebar.file_uploader("**Choose a video...**", type=["mp4", "avi", "mov"], on_change=st.session_state.clear)

vid_col1, vid_col2 = st.columns(2)

if uploaded_file is not None:
    if "uploaded_file_path" not in st.session_state or st.session_state.uploaded_file_path != uploaded_file.name:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            st.session_state.uploaded_file_path = temp_video.name


    if "results" not in st.session_state:
        model = YOLO("yolo11m-pose.pt")
        total_frames = cv2.VideoCapture(st.session_state.uploaded_file_path).get(cv2.CAP_PROP_FRAME_COUNT)
        progress_bar = st.progress(0, text="Processing frame 0 of {}".format(int(total_frames)))  # Initialize progress bar
        
        
        results = []
        for i, result in enumerate(model.track(source=st.session_state.uploaded_file_path, stream=True)):
            results.append(result)
            progress_bar.progress(
                min((i + 1) / total_frames, 1.0),  # Ensure progress doesn't exceed 100%
                text="Processing frame {} of {}".format(i + 1, int(total_frames))  # Update frame number
            )

        st.session_state.results = results
        progress_bar.empty()

        st.success("Video processing complete!", icon="🎉")
    

        

    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    st.session_state.output_video_path = output_video_path

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    
    first_frame = st.session_state.results[0].plot()
    frame_height, frame_width, _ = first_frame.shape
    frame_size = (frame_width, frame_height)

    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    if not out.isOpened():
        raise RuntimeError("Failed to initialize VideoWriter. Check codec compatibility.")
    for result in st.session_state.results:
        annotated_frame = result.plot()
        out.write(annotated_frame)
    out.release()

    convertedVideo = "./testh264.mp4"
    subprocess.call(f"ffmpeg -y -i {output_video_path} -c:v libx264 {convertedVideo}".split(" "))


    with vid_col1:
        st.video(st.session_state.uploaded_file_path)

    with vid_col2:
        st.video(convertedVideo, format="video/mp4")

if "results" in st.session_state:
    results_col1, results_col2 = st.columns(2)
    results = st.session_state.results
    with results_col1:
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)

    if sport == "Sprint Starting Technique":
        player_coords = get_player_coords(player, results)
        scoring, eval_frames = evaluate_sprint_start(player_coords=player_coords)
    elif sport == "Sprint Running Technique":
        player_coords = get_player_coords(player, results)
        scoring, eval_frames = evaluate_sprint_running(player_coords=player_coords)
    elif sport == "Long Jump":
        player_coords = get_player_coords(player, results, True, True)
        scoring, eval_frames = evaluate_long_jump(player_coords=player_coords)
    elif sport == "High Jump":
        player_coords = get_player_coords(player, results, True, True)
        scoring, eval_frames = evaluate_high_jump(player_coords=player_coords)
    elif sport == "Shotput":
        player_coords = get_player_coords(player, results, True, True)
        scoring, eval_frames = evaluate_shot_put(player_coords=player_coords)
    elif sport == "Discus Throw":
        player_coords = get_player_coords(player, results, True, True)
        scoring, eval_frames = evaluate_discus_throw(player_coords=player_coords)
    elif sport == "Javelin Throw":
        player_coords = get_player_coords(player, results)
        scoring, eval_frames = evaluate_javelin_throw(player_coords=player_coords)

    scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
    
    
    st.data_editor(scoring_df,
                    column_config={
        "Criteria": st.column_config.TextColumn(width='large'),
        "Score": st.column_config.NumberColumn(format="%d ⭐")
    }, use_container_width=True)

