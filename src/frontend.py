import streamlit as st
import base64
from sender import send_image 

st.set_page_config(
    layout="wide",
    page_title="Classification de maladies de peau",
)

st.title("Classification de maladies de peau")

image = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])
image_pred = None

should_explain = st.checkbox("Explication", value=False)

type_explain = st.selectbox("Précision de l'explication (une précision importante augmente le temps d'éxecution).",
                            ["Faible", "Moyenne", "Importante"],
                            index=2)
                            
skin_diseases = None
proba_skin_diseases = None

st.write("##")

if st.button("Prediction") and image is not None:
    with st.spinner('Prediction in progress...'):
        img = base64.b64encode(image.read()).decode("utf8")
        success, explain = send_image(img, "http://127.0.0.1:8089/predict", should_explain)
        
        if success:
            skin_diseases = explain[0]
            proba_skin_diseases = explain[1]
            image_pred = explain[2]

st1, st2 = st.columns(2)
                
if image is not None:
    st1.image(image, caption=f"Image : {image.name}", use_column_width=True)
    
if image_pred is not None:
    st2.image(image_pred, caption=f"Image : {skin_diseases}", use_column_width=True)

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
        Les marques jaunes indiquent les zones les plus importantes utilisées par le modèle pour la classification.
                """)