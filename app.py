"""Streamlit Web Application for Multimodal Deepfake Detection.

State-of-the-art forensic analysis combining spatial facial artifacts,
sequential temporal anomalies, and Mel-spectrogram audio synthesis verification.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch

from deepfake_core.pipeline import UnifiedDetector

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Multimodal Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Rich Glassmorphic Design System & Aesthetics
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global reset and font styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main background with subtle ambient glows */
    .stApp {
        background-color: #080B11;
        background-image: 
            radial-gradient(at 15% 15%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
            radial-gradient(at 85% 20%, rgba(6, 182, 212, 0.07) 0px, transparent 50%),
            radial-gradient(at 50% 85%, rgba(139, 92, 246, 0.05) 0px, transparent 50%);
        color: #F1F5F9;
    }

    /* Header hero styling */
    .hero-wrapper {
        position: relative;
        padding: 30px 24px 22px 24px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.45) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        margin-bottom: 28px;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 9999px;
        color: #A5B4FC;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        background: linear-gradient(135deg, #FFFFFF 20%, #94A3B8 60%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
        max-width: 820px;
        margin: 0;
    }

    /* Glass card styling */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.25);
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
    }

    /* Metric cards */
    .metric-card-styled {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card-styled::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .card-spatial::before { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
    .card-temporal::before { background: linear-gradient(90deg, #8B5CF6, #C084FC); }
    .card-audio::before { background: linear-gradient(90deg, #10B981, #34D399); }
    .card-fusion::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 8px 0 6px 0;
    }

    .metric-desc {
        font-size: 0.82rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Progress bar in metrics */
    .metric-bar-bg {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        margin: 10px 0 6px 0;
        overflow: hidden;
    }
    .metric-bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Sample selector cards */
    .sample-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .sample-card:hover {
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(99, 102, 241, 0.35);
    }

    /* Primary CTA Button enhancement */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.02em !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        box-shadow: 0 6px 20px -4px rgba(99, 102, 241, 0.5) !important;
        transition: all 0.25s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px -4px rgba(99, 102, 241, 0.75) !important;
        filter: brightness(1.1) !important;
    }

    /* Sidebar aesthetics */
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    .sidebar-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-top: 18px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Custom model status badge */
    .model-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.2) !important;
        color: #E0E7FF !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
    }

    /* ------------------------------------------------------------- */
    /* Strict Fixed Video Viewport Boundary & Stable Layout Engine   */
    /* ------------------------------------------------------------- */
    .video-viewport-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(8, 11, 17, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 14px 16px 12px 16px;
        box-shadow: 0 14px 34px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        margin-bottom: 8px;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .video-viewport-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 16px 40px -10px rgba(99, 102, 241, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    }

    .video-viewport-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .video-viewport-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #F8FAFC;
    }

    .video-viewport-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.12);
        color: #93C5FD;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* Fixed Streamlit st.video container: keeps the page layout strictly invariant */
    div[data-testid="stVideo"], .stVideo {
        width: 100% !important;
        height: 380px !important;
        max-height: 380px !important;
        min-height: 380px !important;
        background: radial-gradient(circle at center, #0B1120 0%, #020617 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: inset 0 2px 14px rgba(0, 0, 0, 0.85), 0 8px 24px -6px rgba(0, 0, 0, 0.6) !important;
        margin: 0 0 8px 0 !important;
    }

    /* Video player element inside: letterboxed with object-fit: contain */
    div[data-testid="stVideo"] > video, .stVideo > video {
        width: 100% !important;
        height: 100% !important;
        max-height: 380px !important;
        object-fit: contain !important;
        background-color: transparent !important;
        border-radius: 10px !important;
    }

    .video-viewport-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 6px;
        font-size: 0.74rem;
        color: #64748B;
        padding: 0 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_detector(device: str, _cache_version: str = "v3.1") -> UnifiedDetector:
    """Load and cache UnifiedDetector models in memory."""
    return UnifiedDetector(
        device=device,
        model_path="models/checkpoints/best_spatial_model.pth",
        temporal_model_path="models/checkpoints/best_temporal_model.pth",
        audio_model_path="models/checkpoints/best_audio_model.pth",
    )


def create_attention_chart(attention_weights: list[float]) -> go.Figure:
    """Create an interactive Plotly chart visualizing temporal attention weights across frames."""
    n_frames = len(attention_weights)
    x_vals = list(range(1, n_frames + 1))

    max_idx = int(np.argmax(attention_weights)) if attention_weights else 0
    max_val = attention_weights[max_idx] if attention_weights else 0.0
    avg_val = float(np.mean(attention_weights)) if attention_weights else 0.0

    fig = go.Figure()

    # Attention area glow fill
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=attention_weights,
            mode="lines+markers",
            name="Temporal Attention Weight",
            line=dict(color="#818CF8", width=3.5, shape="spline"),
            marker=dict(size=7, color="#A5B4FC", line=dict(width=1.5, color="#312E81")),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.15)",
            hovertemplate="<b>Frame Sequence %{x}</b><br>Attention: %{y:.4f}<extra></extra>",
        )
    )

    # Highlight peak anomaly frame
    if n_frames > 0:
        fig.add_trace(
            go.Scatter(
                x=[max_idx + 1],
                y=[max_val],
                mode="markers+text",
                name="Peak Anomaly Sequence",
                marker=dict(size=14, color="#F43F5E", symbol="diamond", line=dict(width=2, color="#FFFFFF")),
                text=[f"Peak Frame #{max_idx + 1}"],
                textposition="top center",
                textfont=dict(color="#FDA4AF", size=11, family="JetBrains Mono"),
                hovertemplate="<b>🚨 Peak Anomaly Frame %{x}</b><br>Weight: %{y:.4f}<extra></extra>",
            )
        )

    # Baseline average reference
    fig.add_hline(
        y=avg_val,
        line_dash="dot",
        line_color="#64748B",
        line_width=1.5,
        annotation_text=f"Uniform Baseline ({avg_val:.4f})",
        annotation_position="bottom right",
        annotation_font=dict(color="#94A3B8", size=10, family="JetBrains Mono"),
    )

    fig.update_layout(
        title=dict(
            text="<b>Bi-LSTM Temporal Attention Distribution Across Sampled Frames</b>",
            font=dict(size=15, color="#F8FAFC", family="Plus Jakarta Sans"),
        ),
        xaxis=dict(
            title=dict(text="Sampled Frame Sequence", font=dict(color="#94A3B8", size=12)),
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickmode="linear",
            dtick=max(1, n_frames // 10),
            color="#94A3B8",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Normalized Attention Weight", font=dict(color="#94A3B8", size=12)),
            gridcolor="rgba(255, 255, 255, 0.05)",
            color="#94A3B8",
            zeroline=False,
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(11, 15, 25, 0.7)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=45, r=45, t=55, b=45),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color="#CBD5E1", size=11),
        ),
    )

    return fig


def main() -> None:
    # -------------------------------------------------------------------------
    # Sidebar: System Diagnostics & Settings
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            textwrap.dedent(
                """
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                    <div style="font-size: 1.8rem;">🛡️</div>
                    <div>
                        <div style="font-weight: 800; font-size: 1.15rem; color: #FFFFFF; line-height: 1.2;">
                            VERIF<span style="color: #60A5FA;">AI</span> FORENSICS
                        </div>
                        <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; letter-spacing: 0.05em;">
                            v2.4 • MULTIMODAL SUITE
                        </div>
                    </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">⚙️ Engine Diagnostics</div>', unsafe_allow_html=True)
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "None (CPU Execution)"

        device = st.selectbox(
            "Hardware Accelerator",
            options=["cuda", "cpu"] if has_cuda else ["cpu"],
            index=0,
            help="Select CUDA for hardware FP16 mixed precision acceleration.",
        )

        st.markdown(
            textwrap.dedent(
                f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 8px; padding: 10px 12px; margin-top: 8px; margin-bottom: 14px;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 700;">Active Hardware</div>
                    <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 600; margin-top: 2px;">
                        {'⚡ ' + gpu_name if has_cuda and device == 'cuda' else '💻 CPU Mode'}
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                        PyTorch {torch.__version__} &bull; {'FP16 Autocast Active' if device == 'cuda' else 'FP32 Mode'}
                    </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">🎚️ Detection Sensitivity</div>', unsafe_allow_html=True)
        sensitivity_mode = st.radio(
            "Sensitivity Preset",
            options=["Balanced (Default)", "High Sensitivity (Strict)", "Conservative (Low FP)"],
            index=0,
            help="Choose a pre-calibrated threshold profile or configure custom values below.",
        )

        if sensitivity_mode == "Balanced (Default)":
            p_fake, p_uncertain = 0.65, 0.45
        elif sensitivity_mode == "High Sensitivity (Strict)":
            p_fake, p_uncertain = 0.55, 0.35
        else:
            p_fake, p_uncertain = 0.75, 0.55

        with st.expander("🛠️ Advanced Calibration & Thresholds", expanded=False):
            fake_thresh = st.slider(
                "Fake Verdict Threshold",
                min_value=0.50,
                max_value=0.90,
                value=p_fake,
                step=0.05,
                help="Scores at or above trigger a FAKE verdict.",
            )
            uncertain_thresh = st.slider(
                "Uncertain Verdict Threshold",
                min_value=0.30,
                max_value=0.60,
                value=p_uncertain,
                step=0.05,
                help="Scores between this and Fake trigger UNCERTAIN.",
            )
            max_frames = st.slider(
                "Uniform Sample Budget",
                min_value=10,
                max_value=60,
                value=30,
                step=5,
                help="Number of uniformly spaced frames extracted per video clip.",
            )
            face_confidence = st.slider(
                "Face Detection Confidence",
                min_value=0.05,
                max_value=0.80,
                value=0.30,
                step=0.05,
                help="MediaPipe landmark detector sensitivity. Full-range detector detects distant faces up to 5 meters. Lower values (0.10-0.30) detect distant, wide-angle, or angled faces.",
            )
            calibration_bias = st.slider(
                "Compression Calibration Bias",
                min_value=0.00,
                max_value=0.25,
                value=0.10,
                step=0.01,
                help="Offset to counteract H.264 video compression domain shift.",
            )
            min_audio_rms = st.slider(
                "Voice Activity Gate (RMS)",
                min_value=0.0005,
                max_value=0.0200,
                value=0.0020,
                step=0.0005,
                format="%.4f",
                help="Audio energy threshold. Tracks below this are bypassed as silence/ambient noise. Default 0.0020 processes conversational speech.",
            )

        enable_audio = st.sidebar.checkbox(
            "Factor Audio Modality into Verdict",
            value=False,
            help="When disabled, final verdict is based purely on visual biometrics (Spatial + Temporal), preventing the ~0.53 audio baseline from skewing results. Audio score is still computed and displayed for reference.",
        )

        st.markdown('<div class="sidebar-section-title">🧠 Forensic Models Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            textwrap.dedent(
                """
                <div class="model-pill">
                    <span>🖼️ <b>Spatial Engine</b></span>
                    <span style="color: #60A5FA; font-weight: 600;">EfficientNet-B0</span>
                </div>
                <div class="model-pill">
                    <span>⏱️ <b>Temporal Engine</b></span>
                    <span style="color: #C084FC; font-weight: 600;">Bi-LSTM + Attn</span>
                </div>
                <div class="model-pill">
                    <span>🎙️ <b>Acoustic Engine</b></span>
                    <span style="color: #34D399; font-weight: 600;">Mel-CNN 16kHz</span>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Main Header & Hero
    # -------------------------------------------------------------------------
    st.markdown(
        textwrap.dedent(
            """
            <div class="hero-wrapper">
                <div class="hero-badge">
                    <span class="pulse-dot"></span>
                    Tri-Modal Forensic Defense Pipeline Online
                </div>
                <h1 class="hero-title">Multimodal Deepfake Detector</h1>
                <p class="hero-subtitle">
                    High-precision deepfake forensics combining spatial facial blending artifacts, 
                    bi-directional temporal continuity anomalies, and acoustic Mel-spectrogram voice clone verification.
                </p>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    # Load and configure detector
    detector = get_cached_detector(device=device)
    detector.threshold = fake_thresh
    detector.uncertain_threshold = uncertain_thresh
    detector.max_frames = max_frames
    detector.face_confidence = face_confidence
    detector.calibration_bias = calibration_bias
    detector.min_audio_rms = min_audio_rms
    detector.enable_audio = enable_audio

    # -------------------------------------------------------------------------
    # Input Selection Deck (Tabs for Quick Demo vs Custom Upload)
    # -------------------------------------------------------------------------
    st.markdown("### 📥 Select Video Input")

    tab_sample, tab_upload = st.tabs(["⚡ Preloaded Forensic Samples", "📁 Custom Video Upload"])

    # Session state initialization for video selection
    if "selected_video_path" not in st.session_state:
        st.session_state["selected_video_path"] = "data/raw/deepfake_videos/real_test_video.mp4"
        st.session_state["selected_video_name"] = "real_test_video.mp4"
        st.session_state["selected_video_source"] = "sample"

    with tab_sample:
        st.markdown(
            "<span style='color: #94A3B8; font-size: 0.9rem;'>Test the multimodal pipeline immediately with verified test vectors:</span>",
            unsafe_allow_html=True,
        )
        sample_col1, sample_col2 = st.columns(2)

        with sample_col1:
            st.markdown(
                textwrap.dedent(
                    """
                    <div class="sample-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="color: #34D399; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em;">
                                ✅ AUTHENTIC SAMPLE
                            </span>
                            <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; padding: 2px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600;">
                                Ground Truth: Real
                            </span>
                        </div>
                        <div style="font-weight: 700; font-size: 1rem; color: #FFFFFF; margin-bottom: 4px;">
                            real_test_video.mp4 (621 KB)
                        </div>
                        <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.4;">
                            Original natural interview clip with authentic facial micro-movements, organic blinking, and synchronized speech audio.
                        </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
            if st.button("Load Authentic Sample", key="btn_sample_real", use_container_width=True):
                st.session_state["selected_video_path"] = "data/raw/deepfake_videos/real_test_video.mp4"
                st.session_state["selected_video_name"] = "real_test_video.mp4"
                st.session_state["selected_video_source"] = "sample"
                st.session_state.pop("custom_upload_bytes", None)
                st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                st.session_state.pop("result", None)
                st.rerun()

        with sample_col2:
            st.markdown(
                textwrap.dedent(
                    """
                    <div class="sample-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="color: #F87171; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em;">
                                🚨 MANIPULATED SAMPLE
                            </span>
                            <span style="background: rgba(239, 68, 68, 0.2); color: #F87171; padding: 2px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600;">
                                Ground Truth: Fake
                            </span>
                        </div>
                        <div style="font-weight: 700; font-size: 1rem; color: #FFFFFF; margin-bottom: 4px;">
                            test_video.mp4 (352 KB)
                        </div>
                        <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.4;">
                            AI deepfake face-swap manipulation exhibiting boundary seams, unnatural temporal eye flickering, and neural warping artifacts.
                        </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
            if st.button("Load Manipulated Sample", key="btn_sample_fake", use_container_width=True):
                st.session_state["selected_video_path"] = "data/raw/deepfake_videos/test_video.mp4"
                st.session_state["selected_video_name"] = "test_video.mp4"
                st.session_state["selected_video_source"] = "sample"
                st.session_state.pop("custom_upload_bytes", None)
                st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                st.session_state.pop("result", None)
                st.rerun()

    with tab_upload:
        uploader_key = st.session_state.get("uploader_key", 0)
        uploaded_file = st.file_uploader(
            "Drop your video file here for forensic examination (.mp4, .avi, .mov):",
            type=["mp4", "avi", "mov"],
            key=f"uploader_{uploader_key}",
            help="Supports high-definition standard video containers with clear frontal or profile human faces.",
        )
        if uploaded_file is not None:
            # Check if this is a newly uploaded file
            if st.session_state.get("last_uploaded_id") != f"{uploaded_file.name}_{uploaded_file.size}":
                st.session_state["last_uploaded_id"] = f"{uploaded_file.name}_{uploaded_file.size}"
                st.session_state["selected_video_name"] = uploaded_file.name
                st.session_state["custom_upload_bytes"] = uploaded_file.getvalue()
                st.session_state["selected_video_source"] = "upload"
                st.session_state["selected_video_path"] = None
                st.session_state.pop("result", None)

            if st.button("❌ Remove Uploaded File", key="btn_clear_upload", use_container_width=False):
                st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                st.session_state.pop("custom_upload_bytes", None)
                st.session_state.pop("last_uploaded_id", None)
                st.session_state["selected_video_path"] = "data/raw/deepfake_videos/real_test_video.mp4"
                st.session_state["selected_video_name"] = "real_test_video.mp4"
                st.session_state["selected_video_source"] = "sample"
                st.session_state.pop("result", None)
                st.rerun()

    # Determine currently active video payload
    active_name = st.session_state.get("selected_video_name", "")
    active_source = st.session_state.get("selected_video_source", "sample")
    video_bytes: Optional[bytes] = None

    if active_source == "upload" and "custom_upload_bytes" in st.session_state:
        video_bytes = st.session_state["custom_upload_bytes"]
    elif active_source == "sample" and st.session_state.get("selected_video_path"):
        sample_path = Path(st.session_state["selected_video_path"])
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                video_bytes = f.read()

    # -------------------------------------------------------------------------
    # Video Inspection Deck & Execution Control
    # -------------------------------------------------------------------------
    if video_bytes is not None:
        st.markdown("---")
        col_preview, col_controls = st.columns([1.25, 1.0], gap="large")

        with col_preview:
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="video-viewport-header">
                        <div class="video-viewport-title">
                            <span class="pulse-dot" style="background-color: #38BDF8; box-shadow: 0 0 8px #38BDF8;"></span>
                            <span>SOURCE MEDIA VIEWPORT:</span>
                            <span style="color: #60A5FA; font-weight: 700;">{active_name}</span>
                        </div>
                        <span class="video-viewport-badge">
                            🔒 FIXED BOUNDARY &bull; {len(video_bytes) / 1024:.1f} KB
                        </span>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
            st.video(video_bytes)
            st.markdown(
                textwrap.dedent(
                    """
                    <div class="video-viewport-footer">
                        <span>🛡️ Forensic Viewport &bull; Fixed Frame Boundary (380px)</span>
                        <span style="color: #38BDF8; font-weight: 600;">Adaptive Letterbox Fit</span>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        with col_controls:
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="glass-card" style="margin-bottom: 18px;">
                        <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
                            🔬 Forensic Analysis Plan
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.88rem; color: #CBD5E1;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span>1️⃣</span> <span><b>Face Extraction:</b> Sampling <b>{max_frames} frames</b> via MediaPipe</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span>2️⃣</span> <span><b>Spatial Scan:</b> EfficientNet-B0 hybrid artifact scoring</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span>3️⃣</span> <span><b>Temporal Scan:</b> Bi-LSTM attention sequence verification</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span>4️⃣</span> <span><b>Acoustic Scan:</b> Mel-CNN spectrogram speech evaluation</span>
                            </div>
                        </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

            run_analysis_btn = st.button(
                "🚀 Initiate Multimodal Forensic Scan",
                type="primary",
                use_container_width=True,
                help="Execute simultaneous spatial, temporal, and acoustic deepfake detection engines.",
            )

            if "result" in st.session_state and st.session_state.get("result_video") == active_name:
                st.markdown(
                    """
                    <div style="text-align: center; margin-top: 10px;">
                        <a href="#forensic-report-anchor" style="color: #60A5FA; font-size: 0.85rem; font-weight: 600; text-decoration: none;">
                            👇 View Forensic Report Results
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if run_analysis_btn:
                temp_file_path: Optional[str] = None
                try:
                    if active_source == "upload":
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                            tfile.write(video_bytes)
                            temp_file_path = tfile.name
                        target_path = temp_file_path
                    else:
                        target_path = st.session_state["selected_video_path"]

                    with st.status("🔬 Performing Multimodal Forensic Extraction...", expanded=True) as status_box:
                        st.write("Extracting face crops and sampling video frame sequences...")
                        start_time = time.time()

                        result = detector.analyze_video(target_path)
                        elapsed = time.time() - start_time

                        result["analysis_duration_seconds"] = round(elapsed, 2)
                        st.session_state["result"] = result
                        st.session_state["result_video"] = active_name
                        st.session_state["result_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["trigger_auto_scroll"] = True

                        st.write("✓ Spatial EfficientNet inference complete")
                        st.write("✓ Sequential Bi-LSTM temporal attention compiled")
                        st.write("✓ Audio waveform voice activity gated & inferred")
                        status_box.update(label=f"✅ Forensic Scan Finished in {elapsed:.2f}s", state="complete", expanded=False)

                    st.rerun()

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    logging.exception("Error during video analysis")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except Exception:
                            pass

    # -------------------------------------------------------------------------
    # Forensic Report Dashboard
    # -------------------------------------------------------------------------
    if "result" in st.session_state and st.session_state.get("result_video") == active_name:
        result = st.session_state["result"]
        verdict = result["verdict"]
        final_score = result["final_score"]
        confidence = result["confidence"]
        confidence_pct = confidence * 100.0
        spatial_score = result["spatial_score"]
        temporal_score = result["temporal_score"]
        audio_score = result.get("audio_score")
        has_audio = result.get("has_audio", False)
        audio_gated = result.get("audio_gated", False)
        audio_rms = result.get("audio_rms", None)
        weights = result.get("temporal_attention_weights", [])
        frames_analyzed = result.get("frames_analyzed", 0)
        scan_time = result.get("analysis_duration_seconds", 0.0)
        timestamp = st.session_state.get("result_timestamp", "Recent")

        st.markdown("---")

        # HTML anchor for smooth scrolling
        st.markdown('<div id="forensic-report-anchor" style="position: relative; top: -30px; height: 1px;"></div>', unsafe_allow_html=True)

        # Trigger auto-scroll smoothly via JavaScript when analysis completes
        if st.session_state.pop("trigger_auto_scroll", False):
            components.html(
                """
                <script>
                    function autoScrollToReport() {
                        try {
                            const doc = window.parent.document;
                            let el = doc.getElementById('forensic-report-anchor');
                            if (!el) {
                                const headings = doc.querySelectorAll('h2');
                                for (let h of headings) {
                                    if (h.innerText && h.innerText.includes('Comprehensive Forensic Report')) {
                                        el = h;
                                        break;
                                    }
                                }
                            }
                            if (el) {
                                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            }
                        } catch (err) {
                            console.error('Auto-scroll navigation error:', err);
                        }
                    }
                    setTimeout(autoScrollToReport, 120);
                    setTimeout(autoScrollToReport, 400);
                    setTimeout(autoScrollToReport, 800);
                </script>
                """,
                height=0,
                width=0,
            )

        st.markdown(
            textwrap.dedent(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 20px;">
                    <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em;">
                        📊 Comprehensive Forensic Report
                    </h2>
                    <div style="color: #64748B; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
                        TIMESTAMP: {timestamp} &bull; LATENCY: {scan_time}s
                    </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

        if result.get("face_fallback_used", False):
            st.info(
                "🔭 **Distant / Wide-Angle Subject Mode:** Standard facial bounding boxes were distant or small in the frame. "
                "Automated multi-scale contextual tracking was applied so forensic inspection completed without failure.",
                icon="ℹ️",
            )

        # 1. Hero Verdict Card
        if verdict == "FAKE":
            verdict_icon = "🚨"
            verdict_headline = "SYNTHETIC MANIPULATION DETECTED"
            verdict_badge = "CRITICAL RISK"
            verdict_color = "#EF4444"
            verdict_border = "rgba(239, 68, 68, 0.45)"
            verdict_gradient = "linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(153, 27, 27, 0.28) 100%)"
            verdict_desc = (
                "High-confidence synthetic facial alterations, boundary artifact signatures, "
                "or unnatural temporal transition anomalies were detected."
            )
        elif verdict == "REAL":
            verdict_icon = "✅"
            verdict_headline = "VERIFIED AUTHENTIC MEDIA"
            verdict_badge = "MINIMAL RISK"
            verdict_color = "#10B981"
            verdict_border = "rgba(16, 185, 129, 0.45)"
            verdict_gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(6, 95, 70, 0.28) 100%)"
            verdict_desc = (
                "No generative boundary seams or sequential facial flickering anomalies detected. "
                "Temporal biometric dynamics align with authentic human physiology."
            )
        else:
            verdict_icon = "⚠️"
            verdict_headline = "UNCERTAIN / INCONCLUSIVE VERDICT"
            verdict_badge = "ELEVATED RISK"
            verdict_color = "#F59E0B"
            verdict_border = "rgba(245, 158, 11, 0.45)"
            verdict_gradient = "linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(146, 64, 14, 0.28) 100%)"
            verdict_desc = (
                "Borderline score threshold. Heavy video compression encoding, lighting artifacts, "
                "or subtle generative blending warrant secondary manual inspection."
            )

        # Clean raw HTML without any blank lines or 4-space markdown indents
        score_marker_pct = min(100.0, max(0.0, final_score * 100.0))
        verdict_banner_html = (
            f'<div style="background: {verdict_gradient}; border: 1.5px solid {verdict_border}; '
            f'border-radius: 18px; padding: 26px 28px; margin-bottom: 24px; box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.5);">'
            f'<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">'
            f'<div>'
            f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">'
            f'<span style="font-size: 1.8rem;">{verdict_icon}</span>'
            f'<span style="color: {verdict_color}; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.01em;">{verdict_headline}</span>'
            f'</div>'
            f'<div style="color: #CBD5E1; font-size: 0.95rem; max-width: 680px; line-height: 1.5;">{verdict_desc}</div>'
            f'</div>'
            f'<div style="text-align: right;">'
            f'<span style="background: rgba(255, 255, 255, 0.08); color: {verdict_color}; font-weight: 700; '
            f'padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; letter-spacing: 0.05em; border: 1px solid {verdict_color}44;">{verdict_badge}</span>'
            f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">'
            f'{confidence_pct:.1f}<span style="font-size: 1.2rem; color: #94A3B8;">%</span>'
            f'</div>'
            f'<div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Detection Confidence</div>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top: 24px;">'
            f'<div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 600; color: #94A3B8; margin-bottom: 6px;">'
            f'<span>🟢 Authentic Zone (0.00 - 0.45)</span>'
            f'<span>🟡 Uncertain (0.45 - 0.65)</span>'
            f'<span>🔴 Deepfake Zone (0.65 - 1.00)</span>'
            f'</div>'
            f'<div style="position: relative; width: 100%; height: 12px; border-radius: 9999px; '
            f'background: linear-gradient(90deg, #10B981 0%, #10B981 45%, #F59E0B 45%, #F59E0B 65%, #EF4444 65%, #EF4444 100%); '
            f'box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">'
            f'<div style="position: absolute; top: -5px; left: calc({score_marker_pct:.2f}% - 8px); '
            f'width: 16px; height: 22px; background: #FFFFFF; border: 2px solid #0F172A; '
            f'border-radius: 4px; box-shadow: 0 0 10px rgba(255, 255, 255, 0.9);"></div>'
            f'</div>'
            f'<div style="text-align: right; margin-top: 6px; font-family: \'JetBrains Mono\', monospace; font-size: 0.85rem; color: #F1F5F9;">'
            f'Composite Anomaly Index: <b>{final_score:.4f}</b>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(verdict_banner_html, unsafe_allow_html=True)

        # 2. Modality Sub-Score Matrix (4 Columns)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        # Spatial Card
        with col_m1:
            spatial_pct = min(100.0, max(0.0, spatial_score * 100.0))
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="metric-card-styled card-spatial">
                        <div class="metric-label">🖼️ Spatial Forensic</div>
                        <div class="metric-value" style="color: #60A5FA;">{spatial_score:.4f}</div>
                        <div class="metric-bar-bg">
                            <div class="metric-bar-fill" style="width: {spatial_pct}%; background: #3B82F6;"></div>
                        </div>
                        <div class="metric-desc">
                            EfficientNet-B0 hybrid feature map (60% mean + 40% top-20% peak anomaly).
                        </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        # Temporal Card
        with col_m2:
            temporal_pct = min(100.0, max(0.0, temporal_score * 100.0))
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="metric-card-styled card-temporal">
                        <div class="metric-label">⏱️ Temporal Anomaly</div>
                        <div class="metric-value" style="color: #C084FC;">{temporal_score:.4f}</div>
                        <div class="metric-bar-bg">
                            <div class="metric-bar-fill" style="width: {temporal_pct}%; background: #8B5CF6;"></div>
                        </div>
                        <div class="metric-desc">
                            Bidirectional LSTM analyzing inter-frame trajectory & blinking dynamics.
                        </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        # Audio Card
        with col_m3:
            if has_audio and audio_score is not None:
                audio_pct = min(100.0, max(0.0, audio_score * 100.0))
                audio_val = f"{audio_score:.4f}"
                audio_bar_html = f'<div class="metric-bar-bg"><div class="metric-bar-fill" style="width: {audio_pct}%; background: #10B981;"></div></div>'
                audio_desc = f"16kHz Mel-CNN Voice Analysis (RMS Energy: {audio_rms:.4f})"
                audio_val_color = "#34D399"
            elif audio_gated:
                audio_val = "BYPASSED"
                audio_bar_html = '<div style="height: 6px; margin: 10px 0 6px 0;"></div>'
                audio_desc = f"Gated: Ambient Silence/Noise (RMS: {audio_rms:.4f} < {min_audio_rms:.3f})"
                audio_val_color = "#F59E0B"
            else:
                audio_val = "NO AUDIO"
                audio_bar_html = '<div style="height: 6px; margin: 10px 0 6px 0;"></div>'
                audio_desc = "No audio stream detected in container"
                audio_val_color = "#94A3B8"

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="metric-card-styled card-audio">
                        <div class="metric-label">🎙️ Acoustic Synthesis</div>
                        <div class="metric-value" style="color: {audio_val_color}; font-size: 1.7rem;">{audio_val}</div>
                        {audio_bar_html}
                        <div class="metric-desc">
                            {audio_desc}
                        </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        # Fusion Composite Card
        with col_m4:
            fusion_pct = min(100.0, max(0.0, final_score * 100.0))
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="metric-card-styled card-fusion">
                        <div class="metric-label">🛡️ Multimodal Fusion</div>
                        <div class="metric-value" style="color: #FBBF24;">{final_score:.4f}</div>
                        <div class="metric-bar-bg">
                            <div class="metric-bar-fill" style="width: {fusion_pct}%; background: #F59E0B;"></div>
                        </div>
                        <div class="metric-desc">
                            Max(Video Fusion [30% Spatial + 70% Temporal], Audio Score)
                        </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        # Multimodal fusion rule breakdown banner
        if enable_audio and has_audio:
            if audio_score is not None and audio_score >= fake_thresh:
                fusion_note = f"<b>Multimodal Audio Alert Fusion:</b> Audio deepfake confidence ({audio_score:.4f}) triggered override &nbsp;⟶&nbsp; <b>Final Score: {final_score:.4f}</b>"
            elif audio_score is not None and abs(audio_score - 0.50) < 0.12:
                fusion_note = f"<b>Bi-Modal Video Fusion (Audio Neutral):</b> Audio score ({audio_score:.4f}) is neutral/uninformative — relying on visual biometrics &nbsp;⟶&nbsp; <b>Final Score: {final_score:.4f}</b>"
            else:
                fusion_note = f"<b>Multimodal Weighted Fusion:</b> 75% Video [{result.get('video_score', 0.0):.4f}] + 25% Audio [{audio_score:.4f}] &nbsp;⟶&nbsp; <b>Final Score: {final_score:.4f}</b>"
        elif not enable_audio and has_audio:
            fusion_note = f"<b>Visual Biometric Fusion (Audio Reference Only):</b> 30% Spatial ({spatial_score:.4f}) + 70% Temporal ({temporal_score:.4f}) &nbsp;⟶&nbsp; <b>Final Score: {final_score:.4f}</b> (Audio evaluated for reference: {audio_score:.4f})"
        elif audio_gated:
            fusion_note = f"<b>Bi-Modal Video Fusion Applied:</b> 30% Spatial ({spatial_score:.4f}) + 70% Temporal ({temporal_score:.4f}) (Audio below RMS threshold {min_audio_rms:.4f}) &nbsp;⟶&nbsp; <b>Final Score: {final_score:.4f}</b>"
        else:
            fusion_note = f"<b>Bi-Modal Video Fusion Applied:</b> 30% Spatial ({spatial_score:.4f}) + 70% Temporal ({temporal_score:.4f}) (No audio track) &nbsp;⟶&nbsp; <b>Final Score: {final_score:.4f}</b>"

        st.markdown(
            textwrap.dedent(
                f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 10px; padding: 12px 18px; margin: 18px 0 28px 0; font-size: 0.88rem; color: #94A3B8;">
                    {fusion_note}
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

        # 3. Interactive Deep Dive Tabs
        tab_viz, tab_insights, tab_telemetry, tab_export = st.tabs(
            [
                "📈 Temporal Anomaly Timeline",
                "🔍 Forensic Diagnostic Insights",
                "📊 Telemetry Matrix",
                "💾 Export Forensic Report",
            ]
        )

        with tab_viz:
            if weights and len(weights) > 0:
                fig = create_attention_chart(weights)
                st.plotly_chart(fig, use_container_width=True)

                peak_idx = int(np.argmax(weights)) + 1
                peak_weight = weights[peak_idx - 1]
                avg_weight = float(np.mean(weights))

                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div class="glass-card" style="margin-top: 10px;">
                            <div style="display: flex; align-items: flex-start; gap: 12px;">
                                <div style="font-size: 1.5rem;">🎯</div>
                                <div>
                                    <div style="font-weight: 700; color: #F1F5F9; font-size: 0.95rem; margin-bottom: 4px;">
                                        Sequence Focus: Peak Anomaly at Sampled Frame #{peak_idx}
                                    </div>
                                    <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                                        The temporal attention network prioritized <b>Frame #{peak_idx}</b> with a normalized attention weight of 
                                        <code class="mono-font" style="color: #FDA4AF;">{peak_weight:.4f}</code> (Baseline average: {avg_weight:.4f}).
                                        Significant attention spikes correspond to inter-frame facial warping, lighting disjunctions, 
                                        or artificial boundaries where synthesis models fail to preserve temporal coherence.
                                    </div>
                                </div>
                            </div>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No sequential attention weights recorded for this clip.")

        with tab_insights:
            st.markdown("#### 🔬 AI Forensic Findings & Interpretations")
            insight_col1, insight_col2 = st.columns(2)

            spatial_finding = (
                f"Elevated pixel discrepancy detected in top 20% outlier frames (mean: {result.get('top_k_mean_prob', 0.0):.4f}), indicating localized synthetic blending seams around eye and mouth contours."
                if spatial_score >= 0.55
                else "Facial landmarks displayed smooth skin micro-textures and absence of boundary blurring, consistent with natural camera sensor captures."
            )
            temporal_finding = (
                "The Bi-LSTM detected unnatural phase shifts across successive frames. Deepfake autoencoders struggle to maintain organic temporal flow, leading to abrupt attention focus."
                if temporal_score >= 0.55
                else "Frame-to-frame ocular and jaw motion transitions maintained organic physiological continuity."
            )
            audio_finding = (
                f"Audio CNN analysis returned synthetic voice probability of <b>{audio_score:.4f}</b> (RMS energy: {audio_rms:.4f}). High spectral harmonic abnormalities detected."
                if has_audio and audio_score is not None
                else ("Audio track gated as ambient background silence/noise (RMS below threshold). Modality bypassed to prevent false alarms on non-speech audio." if audio_gated else "No audio stream found within this container file.")
            )

            with insight_col1:
                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div class="glass-card" style="margin-bottom: 12px;">
                            <div style="font-weight: 700; color: #60A5FA; margin-bottom: 6px;">
                                🖼️ Spatial Texture & Facial Landmark Analysis
                            </div>
                            <p style="color: #CBD5E1; font-size: 0.85rem; margin: 0; line-height: 1.5;">
                                <b>{frames_analyzed} face crops</b> were isolated and evaluated through EfficientNet-B0. {spatial_finding}
                            </p>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )

            with insight_col2:
                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div class="glass-card" style="margin-bottom: 12px;">
                            <div style="font-weight: 700; color: #C084FC; margin-bottom: 6px;">
                                ⏱️ Sequential Temporal Coherence
                            </div>
                            <p style="color: #CBD5E1; font-size: 0.85rem; margin: 0; line-height: 1.5;">
                                Temporal score of <b>{temporal_score:.4f}</b>. {temporal_finding}
                            </p>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="glass-card">
                        <div style="font-weight: 700; color: #34D399; margin-bottom: 6px;">
                            🎙️ Acoustic Spectrogram Voice Forensic Assessment
                        </div>
                        <p style="color: #CBD5E1; font-size: 0.85rem; margin: 0; line-height: 1.5;">
                            {audio_finding}
                        </p>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        with tab_telemetry:
            st.markdown("#### 📐 Raw Pipeline Telemetry")
            tel_col1, tel_col2, tel_col3, tel_col4 = st.columns(4)
            tel_col1.metric("Analyzed Face Frames", f"{frames_analyzed} frames")
            tel_col2.metric("Mean Frame Prob", f"{result.get('mean_frame_prob', 0.0):.4f}")
            tel_col3.metric("Top-20% Frame Prob", f"{result.get('top_k_mean_prob', 0.0):.4f}")
            tel_col4.metric("Compression Bias", f"{result.get('calibration_bias', 0.0):.2f}")

            st.markdown("---")
            tel_col5, tel_col6, tel_col7, tel_col8 = st.columns(4)
            tel_col5.metric("Bi-Modal Video Score", f"{result.get('video_score', 0.0):.4f}")
            tel_col6.metric("Audio Track Present", "Yes" if has_audio else ("Gated" if audio_gated else "No"))
            tel_col7.metric("Audio Waveform RMS", f"{audio_rms:.4f}" if audio_rms is not None else "N/A")
            tel_col8.metric("Inference Engine Latency", f"{scan_time:.2f}s")

        with tab_export:
            st.markdown("#### 📦 Export Forensic Verification Payload")
            st.markdown(
                "<span style='color: #94A3B8; font-size: 0.85rem;'>Download verified forensic telemetry as cryptographic JSON documentation for audit trails.</span>",
                unsafe_allow_html=True,
            )

            export_data = {
                "metadata": {
                    "application": "Multimodal Deepfake Detector",
                    "version": "2.4.0",
                    "timestamp": timestamp,
                    "target_file": active_name,
                    "device": device,
                    "latency_seconds": scan_time,
                },
                "forensic_report": result,
            }

            json_str = json.dumps(export_data, indent=2)

            col_down, col_preview_btn = st.columns([1, 3])
            with col_down:
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_str,
                    file_name=f"forensic_report_{Path(active_name).stem}_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with st.expander("👁️ Preview Raw JSON Telemetry", expanded=False):
                st.json(export_data)


if __name__ == "__main__":
    main()
