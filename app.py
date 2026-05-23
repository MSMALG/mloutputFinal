import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from groq import Groq
from dotenv import load_dotenv
import os
import gdown

# Loading environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page config 
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# CSS 
st.markdown("""
<style>
.msg-bot {
    background: #e8f5e9;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    font-size: 13px;
    color: #1b5e20;
    max-width: 90%;
    line-height: 1.5;
    margin-bottom: 8px;
}
.msg-user {
    background: #e3f2fd;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    font-size: 13px;
    color: #0d47a1;
    max-width: 90%;
    margin-left: auto;
    line-height: 1.5;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# Load all models and assets with caching to speed up app performance
@st.cache_resource
def load_dl_model():
    return tf.keras.models.load_model("best_model.keras")

#@st.cache_resource
#def load_classical_assets():
 #   return {
  #      "svm":    joblib.load("svm_model.pkl"),
   #     "rf":     joblib.load("rf_model.pkl"),
    #    "scaler": joblib.load("hog_scaler.pkl"),
     #   "pca":    joblib.load("hog_pca.pkl"),
      #  "le":     joblib.load("label_encoder.pkl"),
    #}

@st.cache_resource
def load_classical_assets():
    rf_filename = "rf_model.pkl"
    
    # Automatically downloads the model from Google Drive if it isn't in the deployment workspace yet
    if not os.path.exists(rf_filename):
        with st.spinner("Downloading Random Forest Model from Google Drive... Please wait."):
            google_drive_id = '1oqtjMZw0L5QyJNeuOUMB8dOPcs10zw_J/view?usp=drive_link'
            url = f'https://google.com{google_drive_id}'
            gdown.download(url, rf_filename, quiet=False)
            
    return {
        "svm": joblib.load("svm_model.pkl"),
        "rf": joblib.load(rf_filename),
        "scaler": joblib.load("hog_scaler.pkl"),
        "pca": joblib.load("hog_pca.pkl"),
        "le": joblib.load("label_encoder.pkl"),
    }

@st.cache_data
def load_meta():
    return {
        "class_names": joblib.load("class_names.pkl"),
        "dl_acc":      joblib.load("dl_accuracy.pkl"),
        "svm_acc":     joblib.load("svm_accuracy.pkl"),
        "rf_acc":      joblib.load("rf_accuracy.pkl"),
    }

meta     = load_meta()
assets   = load_classical_assets()
dl_model = load_dl_model()

IMG_SIZE = 224
HOG_SIZE = 64

# Session state init 
if "chat_open"     not in st.session_state: st.session_state.chat_open     = False
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "last_label"    not in st.session_state: st.session_state.last_label    = None
if "last_image"    not in st.session_state: st.session_state.last_image    = None
if "advice_loaded" not in st.session_state: st.session_state.advice_loaded = False

# HOG feature extraction 
def get_hog_features(pil_img):
    """Extract HOG features — same pipeline used during training."""
    img  = pil_img.convert("RGB").resize((HOG_SIZE, HOG_SIZE))
    gray = rgb2gray(np.array(img))
    feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")
    feat_scaled = assets["scaler"].transform([feat])
    feat_pca    = assets["pca"].transform(feat_scaled)
    return feat_pca

# Prediction function for both DL and classical models
def predict(pil_img, model_key):
    if model_key == "efficientnet":
        # EfficientNet requires its own preprocessing 
        # NOT dividing by 255 as preprocess_input handles that internally
        img_arr = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        img_arr = preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        probs   = dl_model.predict(img_arr, verbose=0)[0]
        idx     = np.argmax(probs)
        label   = meta["class_names"][idx]
        conf    = probs[idx]
    else:
        feat  = get_hog_features(pil_img)
        clf   = assets["svm"] if model_key == "svm" else assets["rf"]
        idx   = clf.predict(feat)[0]
        probs = clf.predict_proba(feat)[0]
        label = assets["le"].inverse_transform([idx])[0]
        conf  = probs[idx]
    return label, conf

#  Parse class label into plant and condition 
def parse_label(label):
    parts = label.replace("___", "__").split("__")
    plant = parts[0].replace("_", " ").title() if len(parts) > 0 else "Unknown"
    cond  = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
    return plant, cond

#  Groq response 
def ask_groq(user_message, disease_label):
    """
    Send disease label + user message to Llama 3 via Groq.
    Groq doesn't support image input — the disease name from our
    ML model provides all the context needed for accurate advice.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        plant, condition = parse_label(disease_label)
        is_healthy = "healthy" in condition.lower()

        system_prompt = f"""You are an expert agricultural assistant and plant pathologist.
A machine learning model has analyzed a leaf image and predicted the following:
- Plant: {plant}
- Condition: {'Healthy' if is_healthy else condition}
- Full label: {disease_label}

Your role:
- If diseased: explain what the disease is, how it spreads, treatment steps, and prevention tips
- If healthy: give care tips for {plant} and describe early warning signs to watch for
- Be concise, friendly, and practical
- Use bullet points for treatment or care steps
- Keep responses under 200 words
- Always relate your advice specifically to {plant}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Sorry, I couldn't get a response right now. Error: {str(e)}"

#  Initial advice when chat opens 
def get_initial_advice(disease_label):
    plant, condition = parse_label(disease_label)
    is_healthy = "healthy" in condition.lower()
    if is_healthy:
        msg = f"The ML model detected that this {plant} plant is healthy. Please give me care tips and what symptoms to watch for."
    else:
        msg = f"The ML model detected {condition} in this {plant} plant. Please explain this disease and tell me how to treat it."
    return ask_groq(msg, disease_label)

# MAIN UI
st.title("🌿 Plant Disease Detector")
st.markdown(
    "Upload a leaf image, select a model, and get an instant diagnosis. "
    "Use the **🌿 Plant Advisor** button for expert treatment advice."
)
st.divider()

#  Model selector 
st.subheader("1 · Choose a Model")

MODEL_OPTIONS = {
    f"🧠 EfficientNetB0  —  Accuracy: {meta['dl_acc']:.2%}  [Deep Learning]": "efficientnet",
    f"📐 SVM (HOG)       —  Accuracy: {meta['svm_acc']:.2%}  [Classical ML]": "svm",
    f"🌲 Random Forest   —  Accuracy: {meta['rf_acc']:.2%}  [Classical ML]":  "rf",
}

chosen_label = st.radio("Select model:", list(MODEL_OPTIONS.keys()), index=0)
model_key    = MODEL_OPTIONS[chosen_label]

descriptions = {
    "efficientnet": "**EfficientNetB0** — Deep learning model pretrained on ImageNet, fine-tuned on PlantVillage. Learns visual features automatically from pixels.",
    "svm":          "**SVM (HOG)** — Extracts handcrafted edge/shape descriptors (HOG) then classifies with a Support Vector Machine. Faster but less accurate.",
    "rf":           "**Random Forest (HOG)** — Uses HOG features with an ensemble of decision trees. Fastest but lowest accuracy on this dataset.",
}
st.info(descriptions[model_key])
st.divider()

#  Image upload 
st.subheader("2 · Upload a Leaf Image")
uploaded = st.file_uploader("Choose a leaf image (.jpg or .png)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    st.divider()
    st.subheader("3 · Prediction")

    with st.spinner("Analyzing leaf …"):
        label, confidence = predict(pil_img, model_key)

    plant, condition = parse_label(label)
    is_healthy = "healthy" in condition.lower()

    if is_healthy:
        st.success(f"✅ **{plant}** — Healthy")
    else:
        st.error(f"⚠️ **{plant}** — {condition}")

    st.metric("Confidence", f"{confidence:.1%}")

    with st.expander("Raw class label"):
        st.code(label)

    # Reset chat if new prediction
    if st.session_state.last_label != label:
        st.session_state.last_label    = label
        st.session_state.last_image    = pil_img
        st.session_state.chat_history  = []
        st.session_state.advice_loaded = False

    st.divider()
    st.caption(
        "⚠️ Note: Trained on lab-condition images (PlantVillage). "
        "Outdoor photos may give less reliable results due to domain shift."
    )

#  Plant Advisor 
if st.session_state.last_label and GROQ_API_KEY:

    st.divider()

    btn_label = "🌿 Close Plant Advisor" if st.session_state.chat_open else "🌿 Open Plant Advisor"
    if st.button(btn_label, use_container_width=True):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

    if st.session_state.chat_open:
        st.markdown("### 🌿 Plant Advisor")
        st.caption("Powered by Llama 3.3 70B via Groq — ask anything about your plant")

        # Load initial advice only on first open
        if not st.session_state.advice_loaded:
            with st.spinner("🌿 Analyzing your plant …"):
                initial_advice = get_initial_advice(st.session_state.last_label)
            st.session_state.chat_history.append({
                "role": "bot",
                "text": initial_advice
            })
            st.session_state.advice_loaded = True
            st.rerun()

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "bot":
                st.markdown(f"<div class='msg-bot'>{msg['text']}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='msg-user'>{msg['text']}</div>",
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # User input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask something about this plant …",
                placeholder="e.g. What pesticide should I use?"
            )
            submitted = st.form_submit_button("Send ➤")

        if submitted and user_input.strip():
            st.session_state.chat_history.append({
                "role": "user",
                "text": user_input
            })
            with st.spinner("Thinking …"):
                reply = ask_groq(
                    user_input,
                    st.session_state.last_label
                )
            st.session_state.chat_history.append({
                "role": "bot",
                "text": reply
            })
            st.rerun()

elif not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not found in .env file — chat advisor disabled.")