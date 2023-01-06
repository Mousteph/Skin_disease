import streamlit as st
import base64
from sender import send_image 

st.set_page_config(
    layout="wide",
    page_title="AXI Technologie | EPITA",
)

image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

should_explain = st.checkbox("Should explain", value=False)

if st.button("Predict") and image is not None:
    with st.spinner('Prediction in progress...'):
        img = base64.b64encode(image.read()).decode("utf8")
        success, explain = send_image(img, "http://127.0.0.1:8089/predict", should_explain)
        
        if success:
            st.write(f"prediction: {explain[0]} - {explain[1]}")
            if explain[2] is not None:
                st.image(explain[2])
        
