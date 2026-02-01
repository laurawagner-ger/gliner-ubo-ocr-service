import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from modelscope import snapshot_download

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def _strip_markdown(md: str) -> str:
    text = re.sub(r"```.*?```", "", md, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)  # images
    text = re.sub(r"\[[^\]]*\]\([^)]+\)", "", text)   # links
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def build_docling_converter_with_rapidocr(
    models_repo_id: str = "RapidAI/RapidOCR",
    cache_dir: Optional[str] = None,
) -> DocumentConverter:
    """
    Docling PDF converter with RapidOCR (ONNX models via ModelScope cache).
    """
    download_path = snapshot_download(repo_id=models_repo_id, cache_dir=cache_dir)

    det_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv5", "det", "ch_PP-OCRv5_server_det.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv5", "rec", "ch_PP-OCRv5_rec_server_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
    )

    ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
    )
    pipeline_options = PdfPipelineOptions(ocr_options=ocr_options)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def extract_pdf_text_with_docling(pdf_bytes: bytes, filename: str = "upload.pdf") -> Tuple[str, str]:
    """
    Returns (markdown, plain_text)
    """
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / filename
        p.write_bytes(pdf_bytes)

        converter = build_docling_converter_with_rapidocr()
        result = converter.convert(source=p)

        md = result.document.export_to_markdown()
        plain = _strip_markdown(md)
        return md, plain
