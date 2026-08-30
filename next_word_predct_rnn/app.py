import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------------------------
# 💻 Page Configurations & Visual Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS injected directly to build a premium interface
st.markdown("""
    <style>
    /* Main body background refinement */
    .stApp {
        background-color: #0e1117;
    }
    /* Main Header Styling */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 5px;
    }
    /* Text Input Field Styling */
    .stTextInput>div>div>input {
        background-color: #1a1c23 !important;
        color: #FFFFFF !important;
        border: 2px solid #30363d !important;
        border-radius: 10px !important;
        padding: 12px 15px !important;
        font-size: 16px !important;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 10px rgba(79, 70, 229, 0.5) !important;
    }
    /* Prediction Output Box Styling */
    .prediction-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        border: 1px solid #4338ca;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .prediction-label {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #a5b4fc;
        margin-bottom: 5px;
    }
    .prediction-word {
        font-size: 32px;
        font-weight: 800;
        color: #38bdf8;
        font-family: 'Courier New', monospace;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 📦 Optimized Resource Loading (With Cloud Path Support)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    import os
    
    # Smart path mapping for Streamlit Cloud subfolders
    base_path = "next_word_predct_rnn" if os.path.exists("next_word_predct_rnn") else ""
    
    model_path = os.path.join(base_path, "lstm_model_rnn_new.h5")
    tokenizer_path = os.path.join(base_path, "tokenizer_rnn.pkl")
    max_len_path = os.path.join(base_path, "max_len_rnn.pkl")
    
    try:
        model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        with open(max_len_path, "rb") as f:
            max_len = pickle.load(f)
        return model, tokenizer, max_len
    except Exception as e:
        st.error(f"⚠️ Error loading model resources: {e}")
        return None, None, None


# ------------------------------------------------------------------------------
# 🧠 Core Prediction Logic
# ------------------------------------------------------------------------------
def predict_next_word(text):
    if not model or not tokenizer:
        return "Model Error"
        
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "💡 [No word match]"

# ------------------------------------------------------------------------------
# 🎨 User Interface (UI)
# ------------------------------------------------------------------------------

# Hero Section
st.markdown("<h1>🧠 Next Word Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 16px; margin-bottom: 30px;'>An advanced recurrent neural network (LSTM) engine designed to predict linguistic sequences dynamically.</p>", unsafe_allow_html=True)

# Main Application Layout Container
with st.container():
    # User text input block
    user_input = st.text_input(
        "✍️ Sequence Entry Input:", 
        placeholder="Type a sentence context here (e.g., 'be ready')..."
    )
    
    # Dynamic metric breakdown blocks beneath input box
    col1, col2 = st.columns(2)
    with col1:
        words_count = len(user_input.split()) if user_input.strip() != "" else 0
        st.metric(label="📊 Token Context Count", value=f"{words_count} words")
    with col2:
        char_count = len(user_input)
        st.metric(label="🔠 Character Length", value=f"{char_count} chars")

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # Interactive trigger buttons block using layout columns
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        # Full width primary visual call-to-action button
        predict_clicked = st.button("🚀 Analyze & Predict Sequence", use_container_width=True)

    # Processing and Display Logic execution block
    if predict_clicked:
        if user_input.strip() == "":
            st.toast("⚠️ Context input cannot be empty!", icon="❌")
        else:
            with st.spinner("Processing sequence embedding..."):
                next_word = predict_next_word(user_input)
                
            # Render a premium, futuristic looking card framework for output
            st.markdown(f"""
                <div class="prediction-container">
                    <div class="prediction-label">Inferred Next Token</div>
                    <div class="prediction-word">"{next_word}"</div>
                </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 📝 Technical Pipeline Summary Sidebar (Great info for teachers!)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ Architecture Stack")
    st.info("""
    - **Model Core:** LSTM Recurrent Layer
    - **Optimization:** oneDNN Subsystem
    - **Data Pipeline:** Tokenizer Matrix Embeddings
    - **Frontend Layer:** Streamlit Framework
    """)
    st.markdown("---")
    st.caption("Developed for Academic Presentation")
