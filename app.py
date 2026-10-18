"""Streamlit Web Application for Multimodal Deepfake Detection.

Features:
- Video upload & playback (.mp4, .avi, .mov)
- 3-Way Multimodal Forensic Pipeline (Spatial EfficientNet, Temporal BiLSTM, Audio CNN)
- Dynamic Verdict Banner & Confidence Score
- Modality Sub-Score Breakdown (Spatial, Temporal, Audio)
- Interactive Plotly Temporal Attention Visualization
- Detailed Forensic Telemetry & Raw JSON Export
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from deepfake_core.pipeline import UnifiedDetector

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Multimodal Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich glassmorphism dark aesthetics
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {
        background-color: #0E1117;
    }
    
    /* Header hero styling */
    .hero-container {
        padding: 24px 0 16px 0;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Badge styling */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-green {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-blue {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-purple {
        background-color: rgba(168, 85, 247, 0.2);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_detector(device: str) -> UnifiedDetector:
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

    # Main attention line with gradient fill
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=attention_weights,
            mode="lines+markers",
            name="Attention Weight",
            line=dict(color="#818CF8", width=3, shape="spline"),
            marker=dict(size=6, color="#C084FC", line=dict(width=1, color="#FFFFFF")),
            fill="tozeroy",
            fillcolor="rgba(129, 140, 248, 0.15)",
            hovertemplate="<b>Frame Sequence: %{x}</b><br>Attention Weight: %{y:.4f}<extra></extra>",
        )
    )

    # Highlight peak anomaly frame
    if n_frames > 0:
        fig.add_trace(
            go.Scatter(
                x=[max_idx + 1],
                y=[max_val],
                mode="markers+text",
                name="Peak Anomaly Frame",
                marker=dict(size=14, color="#EF4444", symbol="star", line=dict(width=1, color="#FFFFFF")),
                text=[f"Peak (Frame {max_idx + 1})"],
                textposition="top center",
                textfont=dict(color="#EF4444", size=12, family="sans-serif"),
                hovertemplate="<b>🚨 Peak Anomaly Frame %{x}</b><br>Weight: %{y:.4f}<extra></extra>",
            )
        )

    # Reference baseline line (uniform expectation)
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="#94A3B8",
        line_width=1.5,
        annotation_text=f"Baseline Avg ({avg_val:.4f})",
        annotation_position="bottom right",
        annotation_font=dict(color="#94A3B8", size=10),
    )

    fig.update_layout(
        title=dict(
            text="<b>Temporal Attention Distribution Across Sampled Frames</b>",
            font=dict(size=16, color="#F8FAFC"),
        ),
        xaxis=dict(
            title="Frame Sequence (Uniformly Sampled)",
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickmode="linear",
            dtick=max(1, n_frames // 10),
            color="#94A3B8",
            zeroline=False,
        ),
        yaxis=dict(
            title="Attention Weight",
            gridcolor="rgba(255, 255, 255, 0.08)",
            color="#94A3B8",
            zeroline=False,
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0, 0, 0, 0)",
        ),
    )

    return fig


def main() -> None:
    # --- Sidebar Configuration ---
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")

        has_cuda = torch.cuda.is_available()
        default_device = "cuda" if has_cuda else "cpu"
        device = st.selectbox(
            "Compute Device",
            options=["cuda", "cpu"] if has_cuda else ["cpu"],
            index=0,
            help="Select CUDA for fast GPU inference using FP16 mixed precision.",
        )

        st.markdown("---")
        st.markdown("### 🎚️ Decision Thresholds")
        fake_thresh = st.slider(
            "Fake Classification Threshold",
            min_value=0.50,
            max_value=0.90,
            value=0.65,
            step=0.05,
            help="Final scores at or above this value trigger a FAKE verdict.",
        )
        uncertain_thresh = st.slider(
            "Uncertain Threshold",
            min_value=0.30,
            max_value=0.60,
            value=0.45,
            step=0.05,
            help="Scores between Uncertain and Fake thresholds trigger UNCERTAIN.",
        )
        max_frames = st.slider(
            "Max Frames to Analyze",
            min_value=10,
            max_value=60,
            value=30,
            step=5,
            help="Uniform frame sampling budget per video clip.",
        )
        face_confidence = st.slider(
            "Min Face Confidence",
            min_value=0.30,
            max_value=0.80,
            value=0.50,
            step=0.05,
            help="MediaPipe face detection sensitivity threshold.",
        )
        calibration_bias = st.slider(
            "Compression Calibration Bias",
            min_value=0.00,
            max_value=0.25,
            value=0.10,
            step=0.01,
            help="Penalizes raw spatial probabilities to counteract H.264 video compression domain shift.",
        )
        min_audio_rms = st.slider(
            "Voice Activity Audio RMS Gate",
            min_value=0.005,
            max_value=0.030,
            value=0.015,
            step=0.001,
            format="%.3f",
            help="Clips with audio RMS energy below this gate are treated as ambient silence/noise and bypass the audio CNN.",
        )

        st.markdown("---")
        st.markdown("### 🧠 Forensic Models Active")
        st.markdown(
            """
            <div>
                <span class="status-badge badge-blue">🖼️ EfficientNet-B0 (Spatial)</span><br>
                <span class="status-badge badge-purple">⏱️ Bi-LSTM + Attention (Temporal)</span><br>
                <span class="status-badge badge-green">🎙️ Mel-CNN (Audio)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Header & Hero ---
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Multimodal Deepfake Detector</div>
            <div class="hero-subtitle">
                State-of-the-art forensic analysis combining spatial facial artifacts, sequential temporal anomalies,
                and Mel-spectrogram audio synthesis verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Load Detector ---
    detector = get_cached_detector(device=device)
    # Dynamically apply user thresholds without reloading checkpoints
    detector.threshold = fake_thresh
    detector.uncertain_threshold = uncertain_thresh
    detector.max_frames = max_frames
    detector.face_confidence = face_confidence
    detector.calibration_bias = calibration_bias
    detector.min_audio_rms = min_audio_rms

    # --- Video Selection & Upload Section ---
    st.markdown("### 📤 Input Video")

    sample_videos = {
        "None (Upload my own video)": None,
        "Sample 1: Authentic Interview (real_test_video.mp4)": "data/raw/deepfake_videos/real_test_video.mp4",
        "Sample 2: Synthetic Manipulation (test_video.mp4)": "data/raw/deepfake_videos/test_video.mp4",
    }

    selected_sample = st.selectbox(
        "Choose a pre-loaded sample video or upload your own:",
        options=list(sample_videos.keys()),
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Upload a video file for deepfake forensic inspection (.mp4, .avi, .mov):",
        type=["mp4", "avi", "mov"],
        help="Recommended: standard portrait/landscape videos with detectable human faces.",
    )

    video_source_path: Optional[str] = None
    video_display_bytes = None
    video_name: str = ""

    if uploaded_file is not None:
        video_name = uploaded_file.name
        video_display_bytes = uploaded_file.getvalue()
    elif sample_videos[selected_sample] is not None:
        sample_path = Path(sample_videos[selected_sample])
        if sample_path.exists():
            video_name = sample_path.name
            video_source_path = str(sample_path)
            with open(sample_path, "rb") as f:
                video_display_bytes = f.read()

    # --- Video Preview & Execution ---
    if video_display_bytes is not None:
        col_video, col_action = st.columns([1.5, 1.0])

        with col_video:
            st.markdown(f"**Previewing:** `{video_name}`")
            st.video(video_display_bytes)

        with col_action:
            st.markdown("#### 🔍 Forensic Scan")
            st.info(
                f"Sampling up to **{max_frames} frames** uniformly across the clip.\n\n"
                f"Extracting facial landmarks, temporal trajectory, and audio Mel-frequencies.",
                icon="ℹ️",
            )
            run_btn = st.button("🚀 Run Forensic Analysis", type="primary", use_container_width=True)

        # Trigger analysis
        if run_btn:
            temp_file_path: Optional[str] = None
            try:
                # If uploaded file, write to temporary file
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                        tfile.write(video_display_bytes)
                        temp_file_path = tfile.name
                    target_path = temp_file_path
                else:
                    target_path = video_source_path

                with st.spinner("Analyzing spatiotemporal video frames and audio track..."):
                    result = detector.analyze_video(target_path)
                    st.session_state["result"] = result
                    st.session_state["result_video"] = video_name

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logging.exception("Error during video analysis")
            finally:
                # Clean up temporary file immediately after analysis
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception:
                        pass

    # --- Render Dashboard Results ---
    if "result" in st.session_state and st.session_state.get("result_video") == video_name:
        result = st.session_state["result"]
        verdict = result["verdict"]
        final_score = result["final_score"]
        confidence = result["confidence"]
        confidence_pct = confidence * 100.0
        spatial_score = result["spatial_score"]
        temporal_score = result["temporal_score"]
        audio_score = result.get("audio_score")
        has_audio = result.get("has_audio", False)
        weights = result.get("temporal_attention_weights", [])
        frames_analyzed = result.get("frames_analyzed", 0)

        st.markdown("---")
        st.markdown("## 📊 Forensic Report Dashboard")

        # 1. Verdict Banner
        if verdict == "FAKE":
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.28) 100%);
                            border: 1.5px solid #EF4444; border-radius: 14px; padding: 24px; text-align: center; margin-bottom: 24px;">
                    <h1 style="color: #F87171; margin: 0; font-size: 2.3rem; letter-spacing: 1px;">🚨 DEEPFAKE DETECTED</h1>
                    <p style="color: #FECACA; font-size: 1.25rem; font-weight: 600; margin: 8px 0 0 0;">
                        Confidence: {confidence_pct:.1f}% &nbsp;|&nbsp; Final Forensic Score: {final_score:.4f}
                    </p>
                    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 6px;">
                        Strong biometric manipulation detected across facial texture or temporal frame transitions.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif verdict == "REAL":
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.28) 100%);
                            border: 1.5px solid #10B981; border-radius: 14px; padding: 24px; text-align: center; margin-bottom: 24px;">
                    <h1 style="color: #34D399; margin: 0; font-size: 2.3rem; letter-spacing: 1px;">✅ AUTHENTIC / REAL</h1>
                    <p style="color: #A7F3D0; font-size: 1.25rem; font-weight: 600; margin: 8px 0 0 0;">
                        Confidence: {confidence_pct:.1f}% &nbsp;|&nbsp; Final Forensic Score: {final_score:.4f}
                    </p>
                    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 6px;">
                        No anomalous facial distortion or synthetic sequence discontinuities detected.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.28) 100%);
                            border: 1.5px solid #F59E0B; border-radius: 14px; padding: 24px; text-align: center; margin-bottom: 24px;">
                    <h1 style="color: #FBBF24; margin: 0; font-size: 2.3rem; letter-spacing: 1px;">⚠️ UNCERTAIN / INCONCLUSIVE</h1>
                    <p style="color: #FDE68A; font-size: 1.25rem; font-weight: 600; margin: 8px 0 0 0;">
                        Confidence: {confidence_pct:.1f}% &nbsp;|&nbsp; Final Forensic Score: {final_score:.4f}
                    </p>
                    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 6px;">
                        Borderline scores. Severe video compression artifacts or subtle generation noise warrant manual review.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Modality Sub-Score Breakdown (3 Columns)
        st.markdown("### 🧩 Modality Sub-Score Breakdown")
        col_s, col_t, col_a = st.columns(3)

        with col_s:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span style="color: #94A3B8; font-size: 0.9rem; font-weight: 600;">🖼️ SPATIAL FORENSIC SCORE</span>
                    <h2 style="color: #60A5FA; margin: 8px 0 4px 0; font-size: 2rem;">{spatial_score:.4f}</h2>
                    <p style="color: #64748B; font-size: 0.85rem; margin: 0;">
                        EfficientNet-B0 Hybrid (60% Mean + 40% Top-20% Worst Frames)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_t:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span style="color: #94A3B8; font-size: 0.9rem; font-weight: 600;">⏱️ TEMPORAL ANOMALY SCORE</span>
                    <h2 style="color: #C084FC; margin: 8px 0 4px 0; font-size: 2rem;">{temporal_score:.4f}</h2>
                    <p style="color: #64748B; font-size: 0.85rem; margin: 0;">
                        Bidirectional LSTM with Dynamic Temporal Attention Pooling
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_a:
            audio_gated = result.get("audio_gated", False)
            audio_rms = result.get("audio_rms", None)
            if has_audio and audio_score is not None:
                audio_val_str = f"{audio_score:.4f}"
                audio_sub_str = f"AudioForensicCNN on 16kHz Mel-Spectrogram (RMS: {audio_rms:.4f})" if audio_rms is not None else "AudioForensicCNN on 16kHz Mel-Spectrogram"
                audio_color = "#34D399"
            elif audio_gated:
                audio_val_str = "GATED"
                audio_sub_str = f"Ambient Noise / No Voice (RMS: {audio_rms:.4f} < {min_audio_rms:.3f})"
                audio_color = "#F59E0B"
            else:
                audio_val_str = "N/A"
                audio_sub_str = "No Audio Track Detected in Container"
                audio_color = "#94A3B8"

            st.markdown(
                f"""
                <div class="metric-card">
                    <span style="color: #94A3B8; font-size: 0.9rem; font-weight: 600;">🎙️ AUDIO FORENSIC SCORE</span>
                    <h2 style="color: {audio_color}; margin: 8px 0 4px 0; font-size: 2rem;">{audio_val_str}</h2>
                    <p style="color: #64748B; font-size: 0.85rem; margin: 0;">
                        {audio_sub_str}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Multimodal fusion explanation banner
        if has_audio:
            fusion_formula = "<b>Multimodal Max Fusion:</b> max(Video Score [30% Spatial + 70% Temporal], Audio Score)"
        elif result.get("audio_gated", False):
            fusion_formula = "<b>Bi-Modal Video Fusion:</b> 30% Spatial Score + 70% Temporal Score (Audio Bypassed: Ambient Silence/Noise)"
        else:
            fusion_formula = "<b>Bi-Modal Video Fusion:</b> 30% Spatial Score + 70% Temporal Score (No Audio Track)"

        st.markdown(
            f"""
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px dashed rgba(255, 255, 255, 0.15);
                        border-radius: 8px; padding: 10px 16px; margin: 16px 0 24px 0; font-size: 0.9rem; color: #94A3B8;">
                {fusion_formula} &nbsp;⟶&nbsp; <b>Final Score: {final_score:.4f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Interactive Attention Visualization
        st.markdown("### 📈 Sequential Temporal Attention Anomaly Map")
        if weights and len(weights) > 0:
            fig = create_attention_chart(weights)
            st.plotly_chart(fig, use_container_width=True)

            peak_idx = int(np.argmax(weights)) + 1
            st.info(
                f"💡 **Anomaly Focus:** The temporal attention mechanism assigned peak weight to **Frame {peak_idx}** "
                f"({weights[peak_idx - 1]:.4f}). Spikes above the baseline indicate timestamps with flickering, "
                f"unnatural blinking, or synthetic boundary artifacts.",
                icon="🎯",
            )
        else:
            st.warning("No sequential attention weights recorded for this clip.")

        # 4. Detailed Telemetry & Raw JSON Expander
        with st.expander("📋 Detailed Forensic Telemetry & Raw JSON"):
            st.markdown("#### Frame-Level Statistics")
            sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)
            sub_col1.metric("Frames Analyzed", frames_analyzed)
            sub_col2.metric("Mean Frame Prob", f"{result.get('mean_frame_prob', 0.0):.4f}")
            sub_col3.metric("Top-20% Frame Prob", f"{result.get('top_k_mean_prob', 0.0):.4f}")
            sub_col4.metric("Video Score", f"{result.get('video_score', 0.0):.4f}")

            st.markdown("#### Raw Output Payload")
            st.json(result)


if __name__ == "__main__":
    main()
