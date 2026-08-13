import os
import cv2
import streamlit as st

from pipeline import process_image


# Page Setup
st.set_page_config(
    page_title="AquaVision",
    page_icon="🌊",
    layout="wide"
)


# Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# Title
st.title("🌊 AquaVision")
st.subheader(
    "AI Powered Underwater Image Enhancement & Marine Object Detection"
)
st.divider()


# Upload Image
uploaded_file = st.file_uploader(
    "Upload Underwater Image",
    type=["jpg", "jpeg", "png"]
)


# Process Image
if uploaded_file:

    file_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    with st.spinner("Processing Image..."):

        result = process_image(
            file_path
        )

    st.success(
        "Processing Completed Successfully!"
    )


    # Convert Images
    original = cv2.cvtColor(
        result["original"],
        cv2.COLOR_BGR2RGB
    )

    enhanced = cv2.cvtColor(
        result["enhanced"],
        cv2.COLOR_BGR2RGB
    )

    detected = cv2.cvtColor(
        result["detected_image"],
        cv2.COLOR_BGR2RGB
    )


    # Show Images
    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Original Image")

        st.image(
            original,
            use_container_width=True
        )

    with col2:

        st.subheader("Enhanced Image")

        st.image(
            enhanced,
            use_container_width=True
        )

    with col3:

        st.subheader("Detected Objects")

        st.image(
            detected,
            use_container_width=True
        )

    st.divider()


    # Image Quality Metrics
    st.subheader(
        "📊 Image Quality Metrics"
    )

    metrics = result["metrics"]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Brightness",
            round(
                metrics["after"]["brightness"],
                2
            ),
            f'{metrics["improvement_percent"]["brightness"]}%'
        )

    with c2:

        st.metric(
            "Contrast",
            round(
                metrics["after"]["contrast"],
                2
            ),
            f'{metrics["improvement_percent"]["contrast"]}%'
        )

    with c3:

        st.metric(
            "Sharpness",
            round(
                metrics["after"]["sharpness"],
                2
            ),
            f'{metrics["improvement_percent"]["sharpness"]}%'
        )

    st.divider()


    # Detected Objects
    st.subheader(
        "🐟 Detected Objects"
    )

    detections = result["detections"]

    if len(detections) == 0:

        st.warning(
            "No Objects Detected"
        )

    else:

        table = []

        for obj in detections:

            table.append(
                {
                    "Object": obj["class"].title(),
                    "Confidence (%)": round(
                        obj["confidence"] * 100,
                        1
                    )
                }
            )

        st.table(
            table
        )

    st.divider()


    # Summary
    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Objects Detected",
            len(detections)
        )

    with c2:

        st.metric(
            "Processing Time",
            f'{result["processing_time"]:.2f} sec'
        )