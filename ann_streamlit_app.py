"""
ann_streamlit_app.py  —  Multi-Model ANN Predictor (Streamlit)
===============================================================
Streamlit port of the Tkinter ann_gui.py.

Features:
  • Single prediction  – 8 inputs → 5 DS outputs shown as metric cards
  • Batch prediction   – upload CSV, download results with all 5 DS columns
  • Model info         – per-model architecture, CV metrics, training config
  • fc / fy conversion – user enters ksi; values multiplied by 6.89476 before inference
  • Author box         – "AN"  |  Version 1.0

Run:
    streamlit run ann_streamlit_app.py

Place ann_model_1.pt … ann_model_5.pt in the same folder before launching.
"""

import io, pickle, pathlib, csv, warnings
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import pandas as pd

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DOMAIN CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    "Concrete Compressive Strength",
    "Steel Bar Yield Strength",
    "Average Mass Loss",
    "Pitting Mass Loss",
    "Transverse Mass Loss",
    "Story Height",
    "Bay Width",
    "Slab Thickness",
]
N_FEATURES   = len(FEATURE_NAMES)
OUTPUT_NAMES = ["DS-1", "DS-2", "DS-3", "DS-4", "DS-5"]
MODEL_FILES  = [f"ann_model_{i+1}.pt" for i in range(len(OUTPUT_NAMES))]

KSI_TO_MPA    = 0.145
FC_FY_INDICES = [FEATURE_NAMES.index("Concrete Compressive Strength"), FEATURE_NAMES.index("Steel Bar Yield Strength")]

FIELD_META = {
    "Concrete Compressive Strength":   {"unit": "MPa", "hint": "20 – 30 ksi",    "min": 20.0,   "max": 30.0,  "step": 0.1,  "default": 25.0},
    "Steel Bar Yield Strength":        {"unit": "MPa", "hint": "400 – 530 ksi",  "min": 400.0,   "max": 530.0, "step": 0.1,  "default": 460.0},
    "Average Mass Loss":    {"unit": "%",          "hint": "0 – 30",      "min": 0.0,   "max": 30.0, "step": 0.1,  "default": 15.0},
    "Pitting Mass Loss":    {"unit": "%",          "hint": "0 – 50",      "min": 0.0,   "max": 50.0, "step": 0.1,  "default": 25.0},
    "Transverse Mass Loss": {"unit": "%",          "hint": "0 – 70",      "min": 0.0,   "max": 70.0, "step": 0.1,  "default": 30.0},
    "Story Height":         {"unit": "m",          "hint": "3 – 4 m",     "min": 3.0,   "max": 4.0,  "step": 0.05, "default": 3.0},
    "Bay Width":            {"unit": "m",          "hint": "3.0 – 5.0 m", "min": 3.0,   "max": 5.0,  "step": 0.1,  "default": 4.0},
    "Slab Thickness":       {"unit": "mm",         "hint": "0.12 – 0.2 mm","min": 0.11,  "max": 0.2, "step": 0.01,  "default": 0.15},
}

# DS styling
DS_COLORS = ["#1d6f42", "#1564ad", "#5b21b6", "#b45309", "#9b1c1c"]
DS_TINTS  = ["#ecfdf5", "#eff6ff", "#f5f3ff", "#fffbeb", "#fef2f2"]
DS_BORDER = ["#6ee7b7", "#93c5fd", "#c4b5fd", "#fcd34d", "#fca5a5"]

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class ANN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  LOADERS & PREDICTORS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models…")
def load_all_models(model_dir: str):
    """Load all available checkpoints once; cache across reruns."""
    bundles, loaded_names, errors = [], [], []
    for i, fname in enumerate(MODEL_FILES):
        p = pathlib.Path(model_dir) / fname
        if not p.exists():
            errors.append(f"{fname} not found")
            continue
        try:
            ckpt     = torch.load(str(p), map_location="cpu", weights_only=False)
            cfg      = ckpt["model_config"]
            model    = ANN(cfg["input_dim"], cfg["hidden_layers"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            sx       = pickle.loads(ckpt["scaler_X"])
            sy       = pickle.loads(ckpt["scaler_y"])
            meta     = {
                "cv_metrics"  : ckpt.get("cv_metrics",   {}),
                "train_cfg"   : ckpt.get("train_cfg",    {}),
                "model_config": cfg,
                "file"        : fname,
            }
            bundles.append((model, sx, sy, meta))
            loaded_names.append(OUTPUT_NAMES[i])
        except Exception as e:
            errors.append(f"{fname}: {e}")
    return bundles, loaded_names, errors


def apply_conversion(vals: list) -> list:
    """Multiply fc and fy indices by ksi→MPa factor."""
    out = list(vals)
    for idx in FC_FY_INDICES:
        out[idx] = out[idx] * KSI_TO_MPA
    return out


def predict_single(bundles, vals_converted: list) -> list:
    preds = []
    arr   = np.array(vals_converted, dtype=np.float32).reshape(1, -1)
    for model, sx, sy, _ in bundles:
        with torch.no_grad():
            out = model(torch.tensor(sx.transform(arr))).numpy()
        preds.append(float(sy.inverse_transform(out)[0, 0]))
    return preds


def predict_batch(bundles, X_arr: np.ndarray) -> np.ndarray:
    results = []
    for model, sx, sy, _ in bundles:
        X_sc = sx.transform(X_arr.astype(np.float32))
        with torch.no_grad():
            p = model(torch.tensor(X_sc)).numpy()
        results.append(sy.inverse_transform(p).ravel())
    return np.column_stack(results)  # (N, n_models)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    /* ── page background ─────────────────────────────── */
    .stApp { background-color: #131314; }

    /* ── hide default Streamlit chrome ───────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── top app header bar ──────────────────────────── */
    .app-header {
        background: linear-gradient(90deg, #1c3d6e 0%, #2a5298 100%);
        border-radius: 10px;
        padding: 18px 28px 16px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .app-header-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .app-header-sub {
        font-size: 0.80rem;
        color: #93c5fd;
        margin-top: 4px;
    }
    .app-header-badge {
        background: rgba(255,255,255,0.12);
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 0.78rem;
        color: #bfdbfe;
        text-align: right;
        line-height: 1.6;
    }

    /* ── DS output cards ─────────────────────────────── */
    .ds-card {
        border-radius: 10px;
        padding: 16px 18px 14px 18px;
        border-left: 6px solid;
        margin-bottom: 4px;
    }
    .ds-label {
        font-size: 0.90rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .ds-value {
        font-size: 1.80rem;
        font-weight: 700;
        line-height: 1.1;
    }

    /* ── input section heading ───────────────────────── */
    .section-head {
        font-size: 0.90rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #ffffff;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #e2e8f0;
    }

    /* ── author box ──────────────────────────────────── */
    .author-box {
        background: linear-gradient(90deg, #1c3d6e 0%, #2a5298 100%);
        border-radius: 10px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 20px;
    }
    .author-badge {
        width: 52px;
        height: 52px;
        background: #1971c2;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        flex-shrink: 0;
    }
    .author-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
    }
    .author-version {
        font-size: 0.80rem;
        color: #93c5fd;
        margin-top: 2px;
    }

    /* ── model info card ─────────────────────────────── */
    .info-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .info-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 14px;
    }
    .info-section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-top: 12px;
        margin-bottom: 4px;
        border-top: 1px solid #f1f5f9;
        padding-top: 8px;
    }
    .info-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        padding: 3px 0;
    }
    .info-key   { color: #64748b; }
    .info-val   { font-family: Consolas, monospace; color: #1a202c; font-weight: 600; }
    .info-good  { font-family: Consolas, monospace; color: #276749; font-weight: 600; }

    /* ── sidebar ─────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #1c3d6e !important;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label { color: #93c5fd !important; }

    /* ── metric delta hide ───────────────────────────── */
    [data-testid="stMetricDelta"] { display: none; }

    /* ── tab styling ─────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #e8edf2;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 600;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1864ab !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* ── feature chip ────────────────────────────────── */
    .feat-chip {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 0.80rem;
        color: #1a202c;
        margin: 3px;
    }
    .feat-chip span { color: #94a3b8; font-size: 0.74rem; }

    /* ── conversion notice ───────────────────────────── */
    .conv-notice {
        background: #eff6ff;
        border-left: 4px solid #1971c2;
        border-radius: 0 6px 6px 0;
        padding: 8px 12px;
        font-size: 0.80rem;
        color: #1e40af;
        margin-bottom: 14px;
    }

    /* ── dataframe ───────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* ── button overrides ────────────────────────────── */
    .stButton > button {
        border-radius: 7px;
        font-weight: 600;
        padding: 8px 22px;
    }
    .stButton > button[kind="primary"] {
        background: #1971c2;
        border: none;
        color: white;
    }
    .stButton > button[kind="primary"]:hover { background: #1864ab; }
    </style>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PAGE SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_header(loaded_names):
    chips = "  |  ".join(loaded_names)
    st.markdown(f"""
    <div class="app-header">
      <div>
        <div class="app-header-title">&#9638; ANN-DS Predictor</div>
        <div class="app-header-sub">
          {N_FEATURES} Inputs &nbsp;→&nbsp; {len(loaded_names)} Outputs &nbsp;→&nbsp; DS Predictions
        </div>
      </div>
      <div class="app-header-badge">
        {chips}
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_ds_card(ds: str, value, color: str, tint: str, border: str):
    val_str = f"{value:.4f}" if isinstance(value, float) else "—"
    st.markdown(f"""
    <div class="ds-card" style="
        background:{tint};
        border-left-color:{color};">
      <div class="ds-label" style="color:{color};">{ds}</div>
      <div class="ds-value" style="color:{color};">{val_str}</div>
    </div>
    """, unsafe_allow_html=True)


def render_author_box():
    st.markdown("""
    <div class="author-box">
      <div class="author-badge">DS</div>
      <div>
        <div class="author-name">ANN-DS</div>
        <div class="author-version">Version 1.0 | DOI: </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 1 — Single Prediction ─────────────────────────────────────────────────
def tab_single(bundles, loaded_names):
    col_inp, col_out = st.columns([1, 1.3], gap="large")

    # ── INPUT PANEL ──────────────────────────────────────────
    with col_inp:
        st.markdown('<div class="section-head">Input Parameters</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="conv-notice">'
            '&#8505; Enter the input paramters below for damage state prediction in terms of maximum inter-story drift.'
            '</div>',
            unsafe_allow_html=True)

        vals_raw = {}
        # render inputs in two sub-columns for compactness
        c1, c2 = st.columns(2)
        for i, name in enumerate(FEATURE_NAMES):
            meta    = FIELD_META[name]
            col_sel = c1 if i % 2 == 0 else c2
            label   = f"{name}  ({meta['unit']})"
            with col_sel:
                vals_raw[name] = st.number_input(
                    label,
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=float(meta["default"]),
                    step=float(meta["step"]),
                    help=f"Range: {meta['hint']}",
                    key=f"inp_{name}",
                )

        st.markdown("")
        predict_btn = st.button("⚡  Predict All DS", type="primary",
                                use_container_width=True)

    # ── OUTPUT PANEL ─────────────────────────────────────────
    with col_out:
        st.markdown('<div class="section-head">Predicted Damage States</div>',
                    unsafe_allow_html=True)

        if predict_btn:
            vals_list     = [vals_raw[n] for n in FEATURE_NAMES]
            vals_conv     = apply_conversion(vals_list)
            try:
                preds = predict_single(bundles, vals_conv)
                st.session_state["last_preds"] = dict(zip(loaded_names, preds))
                st.session_state["last_vals"]  = vals_raw
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        preds_dict = st.session_state.get("last_preds", {})

        # 3 + 2 layout
        top_ds  = loaded_names[:3]
        bot_ds  = loaded_names[3:]
        top_col = st.columns(len(top_ds)) if top_ds else []
        bot_col = st.columns(len(bot_ds)) if bot_ds else []

        for col, ds in zip(top_col, top_ds):
            ci = OUTPUT_NAMES.index(ds)
            with col:
                render_ds_card(ds, preds_dict.get(ds, "—"),
                               DS_COLORS[ci], DS_TINTS[ci], DS_BORDER[ci])

        for col, ds in zip(bot_col, bot_ds):
            ci = OUTPUT_NAMES.index(ds)
            with col:
                render_ds_card(ds, preds_dict.get(ds, "—"),
                               DS_COLORS[ci], DS_TINTS[ci], DS_BORDER[ci])

        if not preds_dict:
            st.info("Fill in the input parameters and press **Predict All DS**.")

        if preds_dict:
            st.markdown("")
            st.success(
                "✓  " +
                "   ".join(f"{ds}: **{v:.4f}**"
                           for ds, v in preds_dict.items()))

        # author box
        render_author_box()


# ── TAB 2 — Batch Prediction ─────────────────────────────────────────────────
def tab_batch(bundles, loaded_names):
    st.markdown('<div class="section-head">Batch Prediction via CSV</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="conv-notice">'
        '&#8505; CSV must have <b>{}</b> columns in this order: <code>{}</code>. '
        'The first row may be a header. '
        '</div>'.format(N_FEATURES, ", ".join(FEATURE_NAMES)),
        unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload input CSV",
        type=["csv"],
        help=f"CSV with {N_FEATURES} numeric columns (fc, fy, ... Slab Thickness)")

    if uploaded is None:
        # show a template download
        template_io = io.StringIO()
        writer = csv.writer(template_io)
        writer.writerow(FEATURE_NAMES)
        writer.writerow([25.0, 460.0, 10.0, 15.0, 25.0, 3.0, 5.0, 0.14])
        st.download_button(
            "⬇  Download template CSV",
            data=template_io.getvalue().encode(),
            file_name="ds_input_template.csv",
            mime="text/csv")
        return

    try:
        df_in = pd.read_csv(uploaded)
        # drop any extra columns beyond N_FEATURES
        if df_in.shape[1] < N_FEATURES:
            st.error(f"CSV has only {df_in.shape[1]} columns; "
                     f"need at least {N_FEATURES}.")
            return
        X_arr = df_in.iloc[:, :N_FEATURES].values.astype(np.float32)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    # apply conversion
    X_model = X_arr.copy()
    X_model[:, FC_FY_INDICES] *= KSI_TO_MPA

    with st.spinner(f"Running predictions on {len(X_arr)} rows…"):
        preds = predict_batch(bundles, X_model)

    # build result dataframe
    input_cols  = FEATURE_NAMES
    output_cols = loaded_names
    df_input    = pd.DataFrame(X_arr, columns=input_cols)
    df_preds    = pd.DataFrame(preds, columns=output_cols)
    df_result   = pd.concat([df_input, df_preds], axis=1)

    st.success(f"✓  {len(X_arr)} rows processed  →  {len(loaded_names)} outputs predicted")

    # summary metrics
    st.markdown("#### Prediction Statistics")
    stat_cols = st.columns(len(loaded_names))
    for col, ds in zip(stat_cols, loaded_names):
        with col:
            col_data = df_preds[ds]
            st.metric(label=ds,
                      value=f"{col_data.mean():.4f}",
                      help=f"min {col_data.min():.4f}  |  max {col_data.max():.4f}  |  std {col_data.std():.4f}")

    # show table
    st.markdown("#### Results Table")
    # colour the DS output columns
    def _highlight_ds(col):
        ci = OUTPUT_NAMES.index(col.name) if col.name in OUTPUT_NAMES else -1
        if ci >= 0:
            return [f"background-color: {DS_TINTS[ci]}; color: {DS_COLORS[ci]}; font-weight:600"] * len(col)
        return [""] * len(col)

    styled = df_result.style\
        .apply(_highlight_ds, axis=0)\
        .format({c: "{:.5f}" for c in output_cols})\
        .format({c: "{:.5g}"  for c in input_cols})

    st.dataframe(styled, use_container_width=True, height=340)

    # download
    csv_buf = io.StringIO()
    df_result.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇  Download Full Results CSV",
        data=csv_buf.getvalue().encode(),
        file_name="ann_predictions.csv",
        mime="text/csv",
        type="primary")


# ── TAB 3 — Model Info ────────────────────────────────────────────────────────
def tab_info(bundles, loaded_names):

    # ── Feature chips ─────────────────────────────────────────
    st.markdown('<div class="section-head">Input Features  (shared by all models)</div>',
                unsafe_allow_html=True)

    chips_html = ""
    for i, name in enumerate(FEATURE_NAMES):
        unit = FIELD_META[name]["unit"]
        chips_html += (f'<span class="feat-chip">'
                       f'<b>{i+1}.</b> {name}'
                       f' <span>[{unit}]</span></span>')
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model cards ───────────────────────────────────────────
    st.markdown('<div class="section-head">Per-Model Details</div>',
                unsafe_allow_html=True)

    cols = st.columns(len(bundles))

    for col, ds, (model, sx, sy, meta) in zip(cols, loaded_names, bundles):
        ci    = OUTPUT_NAMES.index(ds)
        color = DS_COLORS[ci]
        tint  = DS_TINTS[ci]

        cfg  = meta["model_config"]
        hl   = cfg["hidden_layers"]
        cv   = meta.get("cv_metrics", {})
        tcfg = meta.get("train_cfg",  {})

        with col:
            # card rendered as HTML
            r2_val   = cv.get("r2")
            rmse_val = cv.get("rmse")
            nrm_val  = cv.get("nrmse")

            arch_rows = "".join([
                f'<div class="info-row"><span class="info-key">Input dim</span><span class="info-val">{cfg["input_dim"]}</span></div>',
                f'<div class="info-row"><span class="info-key">Hidden layers</span><span class="info-val">{len(hl)}</span></div>',
                f'<div class="info-row"><span class="info-key">Neurons</span><span class="info-val">{" → ".join(str(h) for h in hl)}</span></div>',
                f'<div class="info-row"><span class="info-key">Output dim</span><span class="info-val">1</span></div>',
                f'<div class="info-row"><span class="info-key">Activation</span><span class="info-val">ReLU</span></div>',
            ])

            def _fmt(v):
                return f"{v:.6f}" if isinstance(v, float) else "N/A"

            cv_rows = "".join([
                f'<div class="info-row"><span class="info-key">R²</span><span class="info-good">{_fmt(r2_val)}</span></div>',
                f'<div class="info-row"><span class="info-key">RMSE</span><span class="info-good">{_fmt(rmse_val)}</span></div>',
                f'<div class="info-row"><span class="info-key">NRMSE</span><span class="info-good">{_fmt(nrm_val)}</span></div>',
            ])

            cfg_rows = "".join(
                f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>'
                for k, v in tcfg.items()
            ) or '<div class="info-row"><span class="info-key">—</span><span class="info-val">N/A</span></div>'

            st.markdown(f"""
            <div class="info-card" style="border-top: 5px solid {color};">
              <div class="info-card-title" style="color:{color};">{ds}</div>

              <div class="info-section-label">Architecture</div>
              {arch_rows}

              <div class="info-section-label">5-Fold CV Metrics</div>
              {cv_rows}

              <div class="info-section-label">Training Config</div>
              {cfg_rows}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SIDEBAR — model directory picker
# ══════════════════════════════════════════════════════════════════════════════
def sidebar_controls():
    with st.sidebar:
        st.markdown("### ⚙ Settings")
        st.markdown("---")

        default_dir = str(pathlib.Path(__file__).parent)
        model_dir   = st.text_input(
            "Model directory",
            value=default_dir,
            help="Folder containing ann_model_1.pt … ann_model_5.pt")

        st.markdown("---")
        st.markdown("**Expected model files:**")
        for f in MODEL_FILES:
            p    = pathlib.Path(model_dir) / f
            icon = "✅" if p.exists() else "❌"
            st.markdown(f"{icon} `{f}`")

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.75rem; color:#93c5fd; line-height:1.6;">'
            'Version 1.0'
            '</div>',
            unsafe_allow_html=True)

    return model_dir


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="ANN-DS Predictor",
        page_icon="🧱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    model_dir = sidebar_controls()

    # load models (cached)
    bundles, loaded_names, errors = load_all_models(model_dir)

    if errors:
        with st.expander("⚠ Model load warnings", expanded=False):
            for e in errors:
                st.warning(e)

    if not bundles:
        st.error("No models could be loaded. "
                 "Check the model directory in the sidebar.")
        st.stop()

    render_header(loaded_names)

    # tabs
    tab1, tab2, tab3 = st.tabs([
        "⚡  Single Prediction",
        "📋  Batch Prediction",
        "ℹ️  Model Info",
    ])

    with tab1:
        tab_single(bundles, loaded_names)
    with tab2:
        tab_batch(bundles, loaded_names)
    with tab3:
        tab_info(bundles, loaded_names)


if __name__ == "__main__":
    main()
