import streamlit as st
import base64
from sender import send_image
import numpy as np

st.set_page_config(
    layout="wide",
    page_title="Classification de maladies de peau",
)

st.title("Classification de maladies de peau")

image = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])
image_pred = None

should_explain = st.checkbox("Explication", value=False)

type_explain = st.selectbox("Précision de l'explication (une précision importante augmente le temps d'éxecution).",
                            ["Low", "Medium", "High"],
                            index=2)
                            
skin_diseases = None
proba_skin_diseases = None

st.write("##")

if st.button("Prédiction") and image is not None:
    with st.spinner('Prédiction en cours...'):
        img = base64.b64encode(image.read()).decode("utf8")
        data = send_image(img, "http://127.0.0.1:8089/predict", should_explain, type_explain)
        
        if data.get("success", False):
            skin_diseases = data.get("prediction")
            proba_skin_diseases = data.get("probability")
            image_pred = np.array(data.get("image")) if should_explain else None
        else:
            st.error("Une erreur est survenue lors de la prédiction.")


st1, st2 = st.columns(2)
                
if image is not None:
    st1.image(image, caption=f"Image : {image.name}")
    
if image_pred is not None:
    st2.image(image_pred, caption=f"Image : {skin_diseases}")

if image is not None and skin_diseases is not None and proba_skin_diseases is not None:
    if not should_explain:
        st.write(f"""
        ### Rapport de classification :
        - Maladie de peau : {skin_diseases}
        - Probabilité : {round(proba_skin_diseases, 3) * 100}%
                """)
    else:
        st.write(f"""
        ### Rapport de classification :
        - Maladie de peau : **{skin_diseases}**
        - Probabilité : {round(proba_skin_diseases, 3) * 100}% 
        Les formes vertes indiquent les zones qui ont été utilisées pour expliquer la décision du modèle.
                """)