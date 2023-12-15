from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import numpy as np

# Setting page layout
st.set_page_config(
    page_title="Nuclei Segmentation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Main page heading
st.title("Nuclei Segmentation")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Nuclei Segmentaion'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

if model_type == 'Nuclei Segmentaion':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Nuclei Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Nuclei Image",
                         use_column_width=True)
        except Exception as ex:
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Nuclei Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Segment'):
                # image_array = np.array(uploaded_image)
                # detected_teeth = helper.detect_teeth(confidence, model, image_array)
                # res = model.predict(uploaded_image, conf=confidence)
                res = model.predict(uploaded_image, show_conf = False, show_labels= False)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Nuclei Image',
                         use_column_width=True)

                class_names = ['Nuclei']

                # undetected_teeth = list(set(class_names) - set(detected_teeth))

                # st.write(f'Detected Teeth: {", ".join(detected_teeth)}')
                # st.write(f'Undetected Teeth: {", ".join(undetected_teeth)}')

                try:
                    with st.expander("Teeth Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

else:
    st.error("Please select a valid source type!")