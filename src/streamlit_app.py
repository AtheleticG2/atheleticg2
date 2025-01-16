import cv2
import numpy as np
import ultralytics
import matplotlib.pyplot as plt
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import time

from sprint_criteria_checks import evaluate_sprint_running, evaluate_sprint_start, get_player_coords
from longjump_criteria_checks import evaluate_long_jump
from highjump_criteria_checks import evaluate_high_jump

import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os
import shutil
import pandas as pd

# Streamlit App
st.title("Athlete Assist")


# Sport Selection
sport = st.sidebar.selectbox(
    "**Which sport would you like to analyze?**",
    ("Sprint Starting Technique", "Sprint Running Technique", "Long Jump", "High Jump"),
)

st.write("**You selected:**", sport)
uploaded_file = st.sidebar.file_uploader("**Choose a video...**", type=["mp4", "avi", "mov"], on_change=st.session_state.clear)

def play_video(results):
    frame = 0
    for result in results:
        annotated_frame = result.plot()
        time.sleep(0.02)
        frame+=1
        image_placeholder.image(annotated_frame, channels="BGR", caption=f"Frame {frame}")


if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.video(temp_video_path)  # Display the original uploaded video

    if "results" not in st.session_state:
        st.write("Processing video...")
        model = YOLO('src/yolo11m-pose.pt')
        st.session_state.results = model.track(source=temp_video_path) 

    else:
        st.write("Using cached results...")

    # Use the cached results for displaying or replaying
    results = st.session_state.results

    image_placeholder = st.empty()
    button_placeholder = st.empty()
    
    play_btn = button_placeholder.button("Play")
    
    if play_btn:
        play_video(results)

    if sport == "Sprint Starting Technique":
        st.write("Evaluating sprint starting technique...")
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)
        player_coords = get_player_coords(player, results)  
        scoring, eval_frames = evaluate_sprint_start(player_coords=player_coords)
        scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
        st.data_editor(scoring_df, column_config={
                                "Criteria": st.column_config.TextColumn(width='large'),
                                "Score": st.column_config.NumberColumn(
                                    format="%d ⭐")})
    
    elif sport == "Sprint Running Technique":
        st.write("Evaluating sprint running technique...")
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)
        player_coords = get_player_coords(player, results)  
        scoring, eval_frames = evaluate_sprint_running(player_coords=player_coords)
        scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
        st.data_editor(scoring_df, column_config={
                                "Criteria": st.column_config.TextColumn(width='large'),
                                "Score": st.column_config.NumberColumn(
                                    format="%d ⭐")})
        st.write(eval_frames)
        
    elif sport == "Long Jump":
        st.write("Evaluating long jump technique...")
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)
        player_coords = get_player_coords(player, results, True, True) # include boxes
        scoring, eval_frames = evaluate_long_jump(player_coords=player_coords)
        scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
        st.data_editor(scoring_df, column_config={
                                "Criteria": st.column_config.TextColumn(width='large'),
                                "Score": st.column_config.NumberColumn(
                                    format="%d ⭐")})
    
    elif sport == "High Jump":
        st.write("Evaluating high jump technique...")
        player = st.number_input("Enter the player ID", min_value=0, max_value=100, value=0)
        player_coords = get_player_coords(player, results, True, True) # include boxes
        scoring, eval_frames = evaluate_high_jump(player_coords=player_coords)
        scoring_df = pd.DataFrame(list(scoring.items()), columns=['Criteria', 'Score'])
        st.data_editor(scoring_df, column_config={
                                "Criteria": st.column_config.TextColumn(width='large'),
                                "Score": st.column_config.NumberColumn(
                                    format="%d ⭐")})
        st.write(eval_frames)
    
    
    os.remove(temp_video_path)

