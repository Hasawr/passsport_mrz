import numpy as np

# NumPy 2.0 polyfill for legacy dependencies
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from demos.audit_dashboard import render_integration_audit
from demos.mrz_comparator import render_mrz_comparison_tab
from services.passport.detector import PassportDetector
from shared.config import get_settings


st.set_page_config(
    page_title="Passport MRZ OCR Service",
    page_icon="🛂",
    layout="wide",
)


@st.cache_resource
def get_detector() -> PassportDetector:
    settings = get_settings()
    return PassportDetector(
        use_gpu=settings.use_gpu,
        save_debug_images=settings.save_ocr_debug_images,
        concurrent_attempts=settings.ocr_concurrent_attempts,
    )


def main() -> None:
    st.title("Passport MRZ Extraction & Comparison Service")
    st.caption("TD3 ICAO Doc 9303 Passport MRZ OCR Reader (2 lines × 44 characters)")

    main_tab, compare_tab, audit_tab = st.tabs(
        ["Passport OCR", "MRZ Comparison (Actual vs Detected)", "API Audit Log"]
    )

    with main_tab:
        render_passport_ocr()

    with compare_tab:
        render_mrz_comparison_tab(get_detector())

    with audit_tab:
        render_integration_audit()


def render_passport_ocr() -> None:
    mode = st.radio("Mode", ["Single Upload", "Batch Upload"], horizontal=True)
    detector = get_detector()

    if mode == "Single Upload":
        uploaded_file = st.file_uploader(
            "Upload Passport Data Page Image",
            type=["png", "jpg", "jpeg", "bmp"],
        )
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.image(uploaded_file, caption="Uploaded Passport", use_container_width=True)

            with col2:
                with TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir) / uploaded_file.name
                    tmp_path.write_bytes(uploaded_file.getvalue())

                    with st.spinner("Processing Passport MRZ..."):
                        output = detector.detect_from_passport(tmp_path)

                    if output.mrz_line1 and output.mrz_line2:
                        st.success("MRZ Lines Extracted Successfully!")
                        st.code(
                            f"Line 1: {output.mrz_line1}\nLine 2: {output.mrz_line2}",
                            language=None,
                        )
                    else:
                        st.warning("Could not read valid MRZ lines from image.")

                    st.metric("Overall Confidence", f"{output.confidence * 100:.1f}%")

                    if output.mrz_result:
                        res = output.mrz_result
                        st.subheader("Parsed Details")
                        details_df = pd.DataFrame(
                            [
                                {"Field": "Passport Number", "Value": res.passport_number or "N/A"},
                                {"Field": "Surname", "Value": res.surname or "N/A"},
                                {"Field": "Given Names", "Value": res.given_names or "N/A"},
                                {"Field": "Nationality", "Value": res.nationality or "N/A"},
                                {"Field": "Date of Birth", "Value": res.date_of_birth or "N/A"},
                                {"Field": "Sex", "Value": res.sex or "N/A"},
                                {"Field": "Expiry Date", "Value": res.expiry_date or "N/A"},
                                {"Field": "Personal Number", "Value": res.personal_number or "N/A"},
                                {"Field": "Passport Number Checksum", "Value": "Valid ✅" if res.checksum_valid else "Invalid ❌"},
                                {"Field": "Composite Checksum", "Value": "Valid ✅" if res.composite_checksum_valid else "Invalid ❌"},
                                {"Field": "Extraction Method", "Value": res.method},
                            ]
                        )
                        st.dataframe(details_df, use_container_width=True, hide_index=True)

                    if output.notes:
                        with st.expander("Processing Notes"):
                            for note in output.notes:
                                st.write(f"- {note}")

    else:
        uploaded_files = st.file_uploader(
            "Upload Passport Data Page Images",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            if st.button("Process Batch", type="primary"):
                results = []
                with TemporaryDirectory() as tmp_dir:
                    progress_bar = st.progress(0)
                    for idx, file in enumerate(uploaded_files):
                        tmp_path = Path(tmp_dir) / file.name
                        tmp_path.write_bytes(file.getvalue())
                        output = detector.detect_from_passport(tmp_path)

                        item = {
                            "File": file.name,
                            "Status": "Success ✅" if output.mrz_line1 else "Failed ❌",
                            "MRZ Line 1": output.mrz_line1 or "N/A",
                            "MRZ Line 2": output.mrz_line2 or "N/A",
                            "Passport Number": output.mrz_result.passport_number if output.mrz_result else "N/A",
                            "Surname": output.mrz_result.surname if output.mrz_result else "N/A",
                            "Given Names": output.mrz_result.given_names if output.mrz_result else "N/A",
                            "Nationality": output.mrz_result.nationality if output.mrz_result else "N/A",
                            "Confidence": f"{output.confidence * 100:.1f}%",
                            "Checksum": "Valid ✅" if (output.mrz_result and output.mrz_result.checksum_valid) else "Invalid ❌",
                        }
                        results.append(item)
                        progress_bar.progress((idx + 1) / len(uploaded_files))

                df = pd.DataFrame(results)
                st.subheader("Batch Results")
                st.dataframe(df, use_container_width=True, hide_index=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV Report",
                    data=csv,
                    file_name="passport_mrz_results.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
