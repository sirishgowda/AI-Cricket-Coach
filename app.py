import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import subprocess
import imageio_ffmpeg


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="AI Cricket Coach",
    page_icon="🏏",
    layout="wide"
)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🏏 AI Cricket Coach")

st.write(
    "Upload a cricket batting or bowling video "
    "to perform AI-based pose tracking and analysis."
)


# -------------------------------------------------
# VIDEO UPLOAD
# -------------------------------------------------

video_file = st.file_uploader(
    "📹 Upload Cricket Video",
    type=["mp4", "mov", "avi", "mkv"]
)


mode = st.selectbox(
    "🏏 Analysis Mode",
    ["Batting", "Bowling"]
)


# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------

if video_file is not None:

    st.success("✅ Video uploaded successfully!")

    # -------------------------------------------------
    # SAVE INPUT VIDEO
    # -------------------------------------------------

    input_path = os.path.join(
        tempfile.gettempdir(),
        "cricket_input_" + video_file.name
    )

    with open(input_path, "wb") as f:
        f.write(video_file.getbuffer())

    # -------------------------------------------------
    # SHOW ORIGINAL VIDEO
    # -------------------------------------------------

    st.subheader("🎥 Original Video")

    st.video(input_path)

    # -------------------------------------------------
    # START ANALYSIS
    # -------------------------------------------------

    if st.button(
        "🏏 Start AI Pose Analysis",
        use_container_width=True
    ):

        st.info("🤖 Loading YOLO11 Pose model...")

        try:

            # Load YOLO11 pose model
            model = YOLO("yolo11n-pose.pt")

        except Exception as e:

            st.error(
                f"Could not load YOLO11 model: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():

            st.error(
                "❌ Could not open the uploaded video."
            )

            st.stop()


        # -------------------------------------------------
        # VIDEO INFORMATION
        # -------------------------------------------------

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0 or fps != fps:
            fps = 30.0


        frame_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        frame_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )


        # -------------------------------------------------
        # TEMP FILES
        # -------------------------------------------------

        raw_output = os.path.join(
            tempfile.gettempdir(),
            "cricket_raw_output.mp4"
        )

        final_output = os.path.join(
            tempfile.gettempdir(),
            "cricket_ai_coach_output.mp4"
        )


        # -------------------------------------------------
        # OPEN VIDEO WRITER
        # -------------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            raw_output,
            fourcc,
            fps,
            (frame_width, frame_height)
        )


        if not writer.isOpened():

            cap.release()

            st.error(
                "❌ Could not create output video."
            )

            st.stop()


        # -------------------------------------------------
        # DISPLAY VIDEO INFORMATION
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "FPS",
            f"{fps:.2f}"
        )

        col2.metric(
            "Frames",
            total_frames
        )

        col3.metric(
            "Mode",
            mode
        )


        st.subheader(
            "🤖 AI Processing"
        )


        progress = st.progress(0)

        status = st.empty()

        frame_count = 0


        # -------------------------------------------------
        # PROCESS VIDEO
        # -------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break


            try:

                results = model.track(
                    frame,
                    persist=True,
                    verbose=False
                )

                annotated_frame = results[0].plot()

            except Exception:

                # If tracking fails for one frame,
                # keep the original frame.
                annotated_frame = frame


            writer.write(
                annotated_frame
            )


            frame_count += 1


            if total_frames > 0:

                percentage = (
                    frame_count / total_frames
                )

                progress.progress(
                    min(percentage, 1.0)
                )

                status.write(
                    f"Processing frame "
                    f"{frame_count} / {total_frames}"
                )


        # -------------------------------------------------
        # CLOSE VIDEO
        # -------------------------------------------------

        cap.release()

        writer.release()


        progress.progress(1.0)

        status.success(
            f"Processed {frame_count} frames successfully."
        )


        # -------------------------------------------------
        # CONVERT MP4 TO H264
        # -------------------------------------------------

        st.info(
            "🔄 Converting output video to browser-compatible format..."
        )


        try:

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


            command = [
                ffmpeg_path,
                "-y",
                "-i",
                raw_output,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                final_output
            ]


            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )


            if result.returncode != 0:

                st.warning(
                    "H.264 conversion failed. "
                    "Trying the original output."
                )

                final_output = raw_output

        except Exception as e:

            st.warning(
                f"Video conversion problem: {e}"
            )

            final_output = raw_output


        # -------------------------------------------------
        # CHECK OUTPUT
        # -------------------------------------------------

        if not os.path.exists(final_output):

            st.error(
                "❌ Output video was not created."
            )

            st.stop()


        file_size = os.path.getsize(
            final_output
        )


        if file_size == 0:

            st.error(
                "❌ Output video is empty."
            )

            st.stop()


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.success(
            "✅ AI pose analysis completed!"
        )


        st.subheader(
            f"🏏 {mode} Analysis Result"
        )


        st.video(
            final_output,
            format="video/mp4"
        )


        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        with open(
            final_output,
            "rb"
        ) as file:

            video_bytes = file.read()


        st.download_button(
            label="⬇️ Download AI Analysis Video",
            data=video_bytes,
            file_name="AI_Cricket_Coach_Analysis.mp4",
            mime="video/mp4",
            use_container_width=True
        )


        # -------------------------------------------------
        # RESULT INFORMATION
        # -------------------------------------------------

        st.subheader(
            "📊 Analysis Summary"
        )


        st.write(
            f"**Analysis Mode:** {mode}"
        )

        st.write(
            f"**Video FPS:** {fps:.2f}"
        )

        st.write(
            f"**Frames Processed:** {frame_count}"
        )

        st.write(
            "**Pose Tracking:** Completed"
        )

        st.info(
            "The current prototype detects and tracks "
            "human pose keypoints. Cricket-specific "
            "technique scoring and advanced coaching "
            "feedback can be added as the next module."
        )