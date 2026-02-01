import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from gliner import GLiNER

from docling_utils import extract_pdf_text_with_docling
from gliner_postprocess import run_gliner_on_text


BASE = Path(__file__).parent

MODEL_OPTIONS = {
    "ubo_gliner_small-v2.1 (21.01.26)": str(BASE / "models" / "ubo_gliner_small-v2.1_21.01.26"),
    "ubo_gliner_base": str(BASE / "models" / "ubo_gliner_base"),
}

st.set_page_config(page_title="PDF → Docling+RapidOCR → GLiNER", layout="wide")
st.title("PDF hochladen → Text extrahieren (Docling+RapidOCR) → GLiNER testen → Download")

with st.sidebar:
    st.header("Modell & Parameter")
    model_choice = st.selectbox("Modell", list(MODEL_OPTIONS.keys()))
    model_path = Path(MODEL_OPTIONS[model_choice])

    threshold = st.slider("GLiNER Threshold", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    spacy_model = st.text_input("spaCy Modell", value="de_core_news_sm")

    st.divider()
    st.subheader("Regel-Optionen")
    steller_same_sentence = st.toggle("Steller-Regel nur im selben Satz", value=True)
    ubo_same_sentence = st.toggle("UBO-Regel nur im selben Satz", value=True)
    not_ubo_same_sentence = st.toggle("NOT_UBO Override nur im selben Satz", value=True)

    steller_window = st.number_input("steller_window", min_value=0, max_value=5000, value=250, step=25)
    not_ubo_window = st.number_input("not_ubo_window", min_value=0, max_value=5000, value=250, step=25)

@st.cache_resource
def load_model(path: str):
    return GLiNER.from_pretrained(path, local_files_only=True)

if not model_path.exists():
    st.error(f"Modell nicht gefunden: {model_path}\n\nLege die Modelle unter ./models/ ab.")
    st.stop()

model = load_model(str(model_path))
st.sidebar.success("Modell geladen.")

pdf = st.file_uploader("PDF hochladen", type=["pdf"])

if not pdf:
    st.info("Bitte ein PDF hochladen.")
    st.stop()

col1, col2 = st.columns(2, gap="large")

with st.spinner("Extrahiere Text mit Docling + RapidOCR …"):
    md, plain = extract_pdf_text_with_docling(pdf.getvalue(), filename=pdf.name)

with col1:
    st.subheader("Extrahierter Text (Plain)")
    st.text_area("Plain", plain, height=420)

with col2:
    st.subheader("Docling Markdown (optional)")
    st.text_area("Markdown", md, height=420)

st.divider()

run_btn = st.button("▶ GLiNER auf extrahiertem Text ausführen", type="primary")

if run_btn:
    if not plain.strip():
        st.warning("Der extrahierte Text ist leer. (OCR/Scan-Qualität?)")
        st.stop()

    with st.spinner("GLiNER läuft + Regeln werden angewendet …"):
        result = run_gliner_on_text(
            model=model,
            text=plain,
            threshold=float(threshold),
            spacy_model=spacy_model,
            steller_window=int(steller_window),
            steller_same_sentence=bool(steller_same_sentence),
            ubo_same_sentence=bool(ubo_same_sentence),
            not_ubo_window=int(not_ubo_window),
            not_ubo_same_sentence=bool(not_ubo_same_sentence),
        )

    st.success(f"Fertig. Predictions: {result['num_preds']}")
    st.subheader("Gruppierte Übersicht (Labels → Snippets)")
    st.text(result["summary_grouped"])

    # DataFrame for quick preview
    preds = result["preds"]
    rows = []
    for p in preds:
        rows.append({
            "label": p["label"],
            "score": p["score"],
            "start": p["start"],
            "end": p["end"],
            "snippet": plain[p["start"]:p["end"]].replace("\n", " "),
        })
    df = pd.DataFrame(rows).sort_values(["label", "score"], ascending=[True, False])

    with st.expander("Tabelle anzeigen"):
        st.dataframe(df, use_container_width=True, height=320)

    # Build ZIP for download: JSON + CSV + grouped summary + extracted text
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("extracted_text.txt", plain)
        z.writestr("docling_markdown.md", md)
        z.writestr("predictions.json", json.dumps(result, ensure_ascii=False, indent=2))
        z.writestr("predictions.csv", df.to_csv(index=False))
        z.writestr("summary_grouped.txt", result["summary_grouped"])

    buf.seek(0)
    st.download_button(
        "⬇ Ergebnisse herunterladen (ZIP)",
        data=buf.read(),
        file_name="pdf_gliner_results.zip",
        mime="application/zip",
    )
