import streamlit as st
from main import process_video

st.title("License Plate Recognition")

# Video uploader
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Button to start processing
if st.button('Process Video') and uploaded_video is not None:
    st.write("Processing...")
    video_path = "input_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Call your main processing function here
    process_video(video_path, "output.csv")

    st.write("Processing completed. Check the results below.")

    # Display the output video with visualizations
    st.video("output_with_visualization.mp4")

    # Button to download results
    with open("interpolated_output.csv", "rb") as file:
        st.download_button(label="Download CSV",
                           data=file,
                           file_name="interpolated_output.csv",
                           mime="text/csv")