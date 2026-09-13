import streamlit as st
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import joblib
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dysarthria Screening", page_icon="🎙️")

@st.cache_resource
def load_artifacts():
    model = joblib.load('svm_final_combined.pkl')
    imputer = joblib.load('imputer_final.pkl')
    scaler = joblib.load('scaler_final.pkl')
    return model, imputer, scaler

model, imputer, scaler = load_artifacts()

def extract_rich_features(path, sr=16000):
    y, sr = librosa.load(path, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1)
    ])

    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rmse = librosa.feature.rms(y=y).mean()

    snd = parselmouth.Sound(path)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    try:
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter = 0
    try:
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer = 0
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
    except Exception:
        hnr = 0
    try:
        formant = call(snd, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)
        f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
    except Exception:
        f1, f2 = 0, 0

    clinical_feats = np.array([jitter, shimmer, hnr, f1, f2, zcr, rmse])
    return np.concatenate([mfcc_feats, clinical_feats])

st.title("🎙️ Dysarthria Speech Screening")
st.write(
    "Upload a short speech recording (.wav) to screen for acoustic markers "
    "associated with dysarthria. This is a screening aid, not a medical diagnosis — "
    "please consult a speech-language pathologist for clinical evaluation."
)

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing speech..."):
        try:
            feat = extract_rich_features("temp_audio.wav").reshape(1, -1)
            feat_imputed = imputer.transform(feat)
            feat_scaled = scaler.transform(feat_imputed)

            pred = model.predict(feat_scaled)[0]
            prob = model.predict_proba(feat_scaled)[0]

            st.subheader("Result")
            if pred == 1:
                st.error(f"Acoustic markers consistent with dysarthria detected "
                          f"(confidence: {prob[1]*100:.1f}%)")
            else:
                st.success(f"No significant dysarthria markers detected "
                            f"(confidence: {prob[0]*100:.1f}%)")

            st.caption(
                "This tool screens for acoustic patterns only and is not a diagnostic "
                "instrument. Results should be confirmed by a qualified clinician."
            )
        except Exception as e:
            st.error(f"Could not process this file: {e}")
