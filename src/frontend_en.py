import streamlit as st
import base64
from sender import send_image
import numpy as np

st.set_page_config(
    layout="wide",
    page_title="Skin Disease Classification",
)

st.title("Skin Disease Classification")

image = st.file_uploader("Select an image", type=["png", "jpg", "jpeg"])
image_pred = None

should_explain = st.checkbox("Explication", value=False)

type_explain = st.selectbox("Explanation precision (high precision increases execution time).",
                            ["Low", "Medium", "High"],
                            index=2)
                            
skin_diseases = None
proba_skin_diseases = None

st.write("##")

if st.button("Prediction") and image is not None:
    with st.spinner('Prediction in progress...'):
        img = base64.b64encode(image.read()).decode("utf8")
        data = send_image(img, "http://127.0.0.1:8089/predict", should_explain, type_explain)
        
        if data.get("success", False):
            skin_diseases = data.get("prediction")
            proba_skin_diseases = data.get("probability")
            image_pred = np.array(data.get("image")) if should_explain else None
        else:
            st.error("An error occurred during the prediction.")


st1, st2 = st.columns(2)
                
if image is not None:
    st1.image(image, caption=f"Image : {image.name}")
    
if image_pred is not None:
    st2.image(image_pred, caption=f"Image : {skin_diseases}")

if image is not None and skin_diseases is not None and proba_skin_diseases is not None:
    if not should_explain:
        st.write(f"""
        ### Classification report:
        - Skin disease: {skin_diseases}
        - Probability: {round(proba_skin_diseases, 3) * 100}%
                """)
    else:
        st.write(f"""
        ### Classification report:
        - Skin disease: **{skin_diseases}**
        - Probability : {round(proba_skin_diseases, 3) * 100}% 
        Green shapes indicate areas that were used to explain the model's decision.
                """)