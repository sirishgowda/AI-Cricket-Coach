import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import subprocess


st.set_page_config(
    page_title="AI Cricket Coach",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 AI Cricket Coach")
st.write(
    "Upload a cricket batting or bowling video "
    "for AI-based pose analysis."
)

# -----------------------------
# Upload
# -----------------------------

video_file = st.file_uploader(
    "Upload Cricket Video",
    type=["mp4", "mov", "avi", "mkv"]
)

mode = st.selectbox(
    "Analysis Mode",
    ["Batting", "Bowling"]
)


if video_file is not None:

    st.success("Video uploaded successfully!")

    # Save uploaded video
    input_path = os.path.join(
        tempfile.gettempdir(),
        "input_cricket_video.mp4"
    )

    with open(input_path, "wb") as f:
        f.write(video_file.getbuffer())

    st.subheader("Original Video")
    st.video(input_path)

    if st.button(
        "🏏 Start AI Pose Analysis",
        use_container_width=True
    ):

        st.info("Loading YOLO11 Pose model...")

        # Load YOLO11 pose model
        model = YOLO("yolo11n-pose.pt")

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("Could not open uploaded video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Temporary AVI output
        avi_path = os.path.join(
            tempfile.gettempdir(),
            "cricket_pose_output.avi"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"MJPG"
        )

        writer = cv2.VideoWriter(
            avi_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            st.error("Could not create output video.")
            cap.release()
            st.stop()

        progress = st.progress(0)

        frame_count = 0

        st.write(f"Video FPS: {fps:.2f}")
        st.write(f"Total frames: {total_frames}")

        # -----------------------------
        # YOLO processing
        # -----------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                verbose=False
            )

            annotated_frame = results[0].plot()

            writer.write(annotated_frame)

            frame_count += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_count / total_frames,
                        1.0
                    )
                )

        cap.release()
        writer.release()

        progress.progress(1.0)

        st.success("YOLO11 pose analysis completed!")

        # -----------------------------
        # Convert AVI → browser MP4
        # -----------------------------

        output_path = os.path.join(
            tempfile.gettempdir(),
            "cricket_pose_output.mp4"
        )

        st.info("Preparing browser-compatible video...")

        try:

            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg_exe,
                "-y",
                "-i",
                avi_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path
            ]

            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            st.success("Video conversion completed!")

            # -----------------------------
            # Final result
            # -----------------------------

            st.subheader(
                f"🏏 {mode} Analysis Result"
            )

            st.video(output_path)

            st.success(
                "AI pose tracking video is ready!"
            )

        except Exception as e:

            st.error(
                "Video conversion failed."
            )

            st.code(str(e))