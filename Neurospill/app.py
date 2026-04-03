import streamlit as st
import torch
import numpy as np
import cv2
import rasterio
import pandas as pd
from model import AttentionUNetPP # Ensure model.py is in your folder

# --- 1. Hardware & Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_trained_model():
    model = AttentionUNetPP(in_channels=3, out_channels=1).to(device)
    model_path = "m3_t2_best_oil_spill_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found.")
    model.eval()
    return model

def preprocess_sar(img_2ch):
    vv = img_2ch[0].astype(np.float32)
    vh = img_2ch[1].astype(np.float32)
    ratio = vv / (vh + 1e-8)
    features = np.stack([vv, vh, ratio], axis=0)
    f_min, f_max = features.min(), features.max()
    if f_max - f_min > 0:
        return (features - f_min) / (f_max - f_min)
    return features * 0.0

# --- 2. Metrics Engine ---
def calculate_advanced_metrics(pred_mask, gt_mask):
    """Calculates pixel-level classification metrics."""
    pred = (pred_mask > 0).flatten()
    gt = (gt_mask > 0).flatten()
    
    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Accuracy": 1.0, "Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0, "IoU": 1.0
        }
    
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    # Using the corrected IoU formula discussed previously
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8) 
    
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1, "IoU": iou
    }

# --- 3. Patch-Wise Verification Engine ---
def predict_with_patch_verification(model, full_img, patch_size=256, confidence=0.5):
    _, H, W = full_img.shape
    full_mask = np.zeros((H, W), dtype=np.uint8)
    model.eval()
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            p_h, p_w = min(patch_size, H - i), min(patch_size, W - j)
            patch = full_img[:, i:i+p_h, j:j+p_w]
            if p_h < patch_size or p_w < patch_size:
                padded = np.zeros((3, patch_size, patch_size), dtype=np.float32)
                padded[:, :p_h, :p_w] = patch
                patch = padded
            input_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()[:p_h, :p_w]
                patch_mask = (prob_map > confidence).astype(np.uint8)
                if np.sum(patch_mask) > 10:
                    full_mask[i:i+p_h, j:j+p_w] = patch_mask
    return full_mask

def analyze_oil_spill(mask):
    kernel = np.ones((7, 7), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    area_pct = (np.sum(clean_mask) / clean_mask.size) * 100
    severity = "NONE" if area_pct == 0 else "LOW" if area_pct < 0.05 else "MEDIUM" if area_pct < 0.20 else "HIGH"
    return clean_mask, severity, round(area_pct, 4)

# --- 4. Dashboard UI ---
st.set_page_config(page_title="Neuro Spill", layout="wide")

# This centers the main title using HTML
st.markdown("<h1 style='text-align: center;'>NEUROSPILL: The Vision That Defends The Ocean</h1>", unsafe_allow_html=True)


c1, c2 = st.columns(2)
with c1:
    uploaded_file = st.file_uploader("Upload SAR Imagery (.tif)", type=["tif"])
with c2:
    gt_file = st.file_uploader("Upload Ground Truth Mask ", type=["tif"])

if uploaded_file:
    model = load_trained_model()
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.99, 0.75)

    with rasterio.open(uploaded_file) as src:
        raw_data = src.read([1, 2])
    
    processed_features = preprocess_sar(raw_data)
    
    with st.spinner("Analyzing..."):
        raw_mask = predict_with_patch_verification(model, processed_features, confidence=conf_thresh)
        final_mask, sev, area = analyze_oil_spill(raw_mask)

    metrics = None
    if gt_file:
        with rasterio.open(gt_file) as gt_src:
            gt_raw = gt_src.read(1)
            if gt_raw.shape != final_mask.shape:
                gt_raw = cv2.resize(gt_raw, (final_mask.shape[1], final_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            metrics = calculate_advanced_metrics(final_mask, gt_raw)

    # --- Intelligence Metrics ---
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detection", "OIL SPILL" if area > 0 else "NO OIL SPILL")
    m2.metric("Coverage", f"{area}%")
    m3.metric("Severity", sev)
    if metrics: m4.metric("IoU Score", f"{metrics['IoU']:.4f}")

    if metrics:
        st.subheader("Statistical Validation")
        col_stats, col_cm = st.columns([1, 1])
        
        with col_stats:
            st.markdown("**Performance Metrics**")
            # Shows Accuracy,Precision,Recall
            stats_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Value": [f"{metrics['Accuracy']:.4f}", f"{metrics['Precision']:.4f}", f"{metrics['Recall']:.4f}", f"{metrics['F1-Score']:.4f}"]
            })
            st.table(stats_df)
            
        with col_cm:
            st.markdown("**Pixel Confusion Matrix**")
            cm_data = [
                [f"TN: {metrics['TN']}", f"FP: {metrics['FP']}"],
                [f"FN: {metrics['FN']}", f"TP: {metrics['TP']}"]
            ]
            st.table(pd.DataFrame(cm_data, columns=["Actual Negative", "Actual Positive"], index=["Pred Negative", "Pred Positive"]))

    # --- Visual Reports ---
    st.subheader("Visual Analysis Report")
    v_cols = st.columns(4 if gt_file else 3)
    vv_norm = (raw_data[0] - raw_data[0].min()) / (raw_data[0].max() - raw_data[0].min() + 1e-8)
    overlay = cv2.cvtColor((vv_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay[final_mask == 1] = [255, 0, 0]
    
    v_cols[0].image(vv_norm, caption="Raw SAR (VV)", use_container_width=True)
    v_cols[1].image(final_mask * 255, caption="Model Prediction", use_container_width=True)
    if gt_file:
        v_cols[2].image((gt_raw > 0) * 255, caption="Ground Truth", use_container_width=True)
        v_cols[3].image(overlay, caption="Analysis Overlay", use_container_width=True)
    else:
        v_cols[2].image(overlay, caption="Analysis Overlay", use_container_width=True)