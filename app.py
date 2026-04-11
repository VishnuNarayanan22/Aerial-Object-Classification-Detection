
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

#Page Config
st.set_page_config(
    page_title="Aerial Object Classifier",
    page_icon="🦅",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('/content/drive/MyDrive/Aerial Object Classification & Detection/models/efficientnet_best.h5')

model = load_model()

#Sidebar
st.sidebar.title(" Model Info")
st.sidebar.markdown("""
**Model:** EfficientNetB0

**Performance:**
| Metric | Score |
|--------|-------|
| Accuracy | 97.73% |
| Precision | 97.85% |
| Recall | 96.81% |
| F1 Score | 97.33% |

**Classes:**
- 🐦 Bird
- 🚁 Drone

**Input Size:** 224 x 224
""")

st.sidebar.divider()
st.sidebar.markdown("""
**Domain:**
Aerial Surveillance,
Wildlife Monitoring,
Security & Defense
""")

# Main UI
st.title("🦅 Aerial Object Classifier")
st.subheader("Bird vs Drone Detection System")
st.markdown("""
This app uses **EfficientNetB0** deep learning model
to classify aerial images as **Bird** or **Drone**
with **97.73% accuracy.**
""")
st.divider()

# Upload
uploaded_file = st.file_uploader(
    "📤 Upload an Aerial Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear aerial image of a bird or drone"
)

if uploaded_file is not None:

    # Show uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

    st.divider()

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array   = np.array(img_resized, dtype=np.float32)
    img_array   = preprocess_input(img_array)
    img_array   = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Analyzing image..."):
        prediction  = model.predict(img_array, verbose=0)[0][0]

    bird_conf  = (1 - prediction) * 100
    drone_conf = prediction * 100

    # Determine result
    if prediction < 0.5:
        label = "🐦 Bird"
        confidence = bird_conf
        color = "green"
        message = "A Bird has been detected in this aerial image!"
    else:
        label = "🚁 Drone"
        confidence = drone_conf
        color = "red"
        message = "A Drone has been detected in this aerial image!"

    # Results
    st.markdown(f"## Prediction: :{color}[{label}]")
    st.markdown(f"*{message}*")

    # Confidence bar
    st.markdown(f"### Confidence: {confidence:.2f}%")
    st.progress(int(confidence) / 100)
    st.divider()

    # Breakdown
    st.markdown("### 📊 Confidence Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="🐦 Bird",
            value=f"{bird_conf:.2f}%",
            delta="Detected!" if prediction < 0.5 else None )

    with col2:
        st.metric(
            label="🚁 Drone",
            value=f"{drone_conf:.2f}%",
            delta="Detected!" if prediction >= 0.5 else None)

    st.divider()

    # Confidence warning
    if confidence >= 90:
        st.success(f"✅ Very High Confidence: {label} detected with {confidence:.2f}%!")
    elif confidence >= 75:
        st.info(f"ℹ️ High Confidence: {label} detected with {confidence:.2f}%")
    else:
        st.warning("⚠️ Low confidence. Please try with a clearer image.")

    # Use case info
    st.divider()
    st.markdown("### 🌍 Real World Application")
    if prediction < 0.5:
        st.markdown("""
        - 🌿 **Wildlife Protection** — Bird detected near wind farms
        - ✈️ **Airport Safety** — Bird strike prevention alert
        - 🔬 **Environmental Research** — Bird population tracking
        """)
    else:
        st.markdown("""
        - 🔒 **Security Alert** — Drone detected in restricted airspace
        - 🛡️ **Defense Surveillance** — Unauthorized drone activity
        - 🚨 **Airspace Safety** — Immediate action recommended
        """)

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome!

    This system can identify:

    | Object | Description |
    |--------|-------------|
    | 🐦 Bird | Natural flying creatures |
    | 🚁 Drone | Unmanned aerial vehicles |

    ### 🚀 How to Use:
    1. Click **Browse files** above
    2. Upload an aerial image (jpg/png)
    3. Get instant prediction with confidence score!

    ### 📌 Use Cases:
    - Wildlife Protection
    - Security & Defense Surveillance
    - Airport Bird-Strike Prevention
    - Environmental Research
    """)
