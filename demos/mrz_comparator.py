import html
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

DEFAULT_SYNTHETIC_DIR = Path(r"C:\Users\Hesen\Desktop\Sinam\passport_mrz_proj_latest\syntetic_images")
DEFAULT_EXCEL_PATH = DEFAULT_SYNTHETIC_DIR / "mrz_of_all.xlsx"


def load_ground_truth_excel(source: Any) -> Dict[str, Dict[str, str]]:
    """
    Load ground truth mapping from Excel file path or file-like stream.
    Returns dict keyed by image filename (e.g. '1.png' -> {'line1': ..., 'line2': ..., ...})
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                return {}
            df = pd.read_excel(path, skiprows=2)
        else:
            # File-like object uploaded in Streamlit
            df = pd.read_excel(source, skiprows=2)

        # Standardize column names
        df.columns = [str(c).strip() for c in df.columns]

        gt_map = {}
        for _, row in df.iterrows():
            img_name = str(row.get("Image Name", "")).strip()
            if not img_name or img_name.lower() in ("nan", "none", ""):
                continue

            # Standardize filename (handle both exact name and basename)
            img_key = Path(img_name).name.lower()
            l1 = str(row.get("MRZ Line 1", "")).strip()
            l2 = str(row.get("MRZ Line 2", "")).strip()

            # Safeguard: normalize 45-character MRZ lines by removing extra filler
            if len(l1) == 45 and l1.endswith("<"):
                l1 = l1[:-1]
            if len(l2) == 45:
                if "<<<<<<<<<<<<<<<" in l2:
                    l2 = l2.replace("<<<<<<<<<<<<<<<", "<<<<<<<<<<<<<<", 1)
                elif l2.endswith("<"):
                    l2 = l2[:-1]

            gt_map[img_key] = {
                "image_name": img_name,
                "full_name": str(row.get("Full Name", "")).strip(),
                "passport_no": str(row.get("Passport No.", "")).strip(),
                "line1": l1,
                "line2": l2,
            }
        return gt_map
    except Exception as e:
        st.error(f"Error loading Ground Truth Excel: {e}")
        return {}


def compare_lines(actual: str, detected: str) -> Dict[str, Any]:
    """
    Compare actual vs detected MRZ line character by character.
    Returns diff html, accuracy percentage, total chars, matches, and mismatches.
    """
    actual = actual or ""
    detected = detected or ""

    max_len = max(len(actual), len(detected), 44)
    matches = 0
    mismatches = 0

    diff_actual_html = []
    diff_detected_html = []

    for i in range(max_len):
        act_char = actual[i] if i < len(actual) else ""
        det_char = detected[i] if i < len(detected) else ""

        esc_act = html.escape(act_char) if act_char else "&nbsp;"
        esc_det = html.escape(det_char) if det_char else "&nbsp;"

        if act_char == det_char and act_char != "":
            matches += 1
            # Green match
            diff_actual_html.append(
                f'<span style="background-color:#1e4620; color:#4CAF50; border:1px solid #2e7d32; font-weight:bold; font-family:monospace; padding:1px 3px; margin:0 1px; border-radius:3px;">{esc_act}</span>'
            )
            diff_detected_html.append(
                f'<span style="background-color:#1e4620; color:#4CAF50; border:1px solid #2e7d32; font-weight:bold; font-family:monospace; padding:1px 3px; margin:0 1px; border-radius:3px;">{esc_det}</span>'
            )
        else:
            mismatches += 1
            # Red mismatch
            title_act = f"Detected: {det_char if det_char else 'MISSING'}"
            title_det = f"Expected: {act_char if act_char else 'EXTRA'}"

            diff_actual_html.append(
                f'<span title="{title_act}" style="background-color:#4a1515; color:#FF5252; border:1px solid #c62828; font-weight:bold; font-family:monospace; padding:1px 3px; margin:0 1px; border-radius:3px;">{esc_act}</span>'
            )
            diff_detected_html.append(
                f'<span title="{title_det}" style="background-color:#4a1515; color:#FF5252; border:1px solid #c62828; font-weight:bold; font-family:monospace; padding:1px 3px; margin:0 1px; border-radius:3px;">{esc_det}</span>'
            )

    accuracy = (matches / max_len * 100.0) if max_len > 0 else 0.0

    return {
        "accuracy": round(accuracy, 2),
        "matches": matches,
        "mismatches": mismatches,
        "max_len": max_len,
        "is_exact_match": actual == detected and actual != "",
        "diff_actual_html": "".join(diff_actual_html),
        "diff_detected_html": "".join(diff_detected_html),
    }


def render_mrz_comparison_tab(detector: Any) -> None:
    st.subheader("🔍 Synthetic Dataset MRZ Comparison (Actual vs Detected)")
    st.markdown(
        "Upload images or run a benchmark against the synthetic dataset to compare "
        "**actual MRZ lines (from Excel)** vs **detected MRZ lines (from OCR engine)**."
    )

    # 1. Ground truth source selection
    excel_file = st.file_uploader(
        "Upload Ground Truth Excel (Optional - defaults to mrz_of_all.xlsx)",
        type=["xlsx", "xls"],
        key="gt_excel_uploader",
    )

    gt_data = load_ground_truth_excel(excel_file if excel_file else DEFAULT_EXCEL_PATH)

    if gt_data:
        st.success(f"Ground Truth loaded successfully ({len(gt_data)} images mapped).")
        with st.expander("📄 View Ground Truth Reference Table"):
            gt_df = pd.DataFrame(list(gt_data.values()))
            st.dataframe(gt_df, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "No ground truth data loaded. Please upload `mrz_of_all.xlsx` or check default path."
        )

    # 2. Mode selection
    mode = st.radio(
        "Comparison Mode",
        ["One-Click Synthetic Benchmark", "Upload & Compare Single Image", "Upload & Compare Batch Images"],
        horizontal=True,
    )

    if mode == "One-Click Synthetic Benchmark":
        render_benchmark_section(detector, gt_data)
    elif mode == "Upload & Compare Single Image":
        render_single_upload_section(detector, gt_data)
    else:
        render_batch_upload_section(detector, gt_data)


def render_benchmark_section(detector: Any, gt_data: Dict[str, Dict[str, str]]) -> None:
    st.info(
        f"Benchmark will process all images in synthetic dataset folder: `{DEFAULT_SYNTHETIC_DIR}`"
    )

    if st.button("🚀 Run Full Synthetic Benchmark", type="primary"):
        if not DEFAULT_SYNTHETIC_DIR.exists():
            st.error(f"Folder not found: {DEFAULT_SYNTHETIC_DIR}")
            return

        image_files = sorted(
            [
                f
                for f in DEFAULT_SYNTHETIC_DIR.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
            ]
        )

        if not image_files:
            st.warning("No image files found in synthetic directory.")
            return

        results = []
        total_l1_acc = 0.0
        total_l2_acc = 0.0
        exact_match_count = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, img_path in enumerate(image_files):
            status_text.text(f"Processing {img_path.name} ({idx+1}/{len(image_files)})...")
            output = detector.detect_from_passport(img_path)

            gt_info = gt_data.get(img_path.name.lower(), {})
            act_l1 = gt_info.get("line1", "")
            act_l2 = gt_info.get("line2", "")

            det_l1 = output.mrz_line1 or ""
            det_l2 = output.mrz_line2 or ""

            comp_l1 = compare_lines(act_l1, det_l1)
            comp_l2 = compare_lines(act_l2, det_l2)

            is_exact = comp_l1["is_exact_match"] and comp_l2["is_exact_match"]
            if is_exact:
                exact_match_count += 1

            total_l1_acc += comp_l1["accuracy"]
            total_l2_acc += comp_l2["accuracy"]

            results.append(
                {
                    "file_path": img_path,
                    "filename": img_path.name,
                    "gt_info": gt_info,
                    "output": output,
                    "act_l1": act_l1,
                    "act_l2": act_l2,
                    "det_l1": det_l1,
                    "det_l2": det_l2,
                    "comp_l1": comp_l1,
                    "comp_l2": comp_l2,
                    "is_exact": is_exact,
                }
            )

            progress_bar.progress((idx + 1) / len(image_files))

        status_text.text("Benchmark complete!")

        # Summary Metrics
        st.subheader("📊 Benchmark Results Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Passports", len(results))
        col2.metric("Exact 2-Line Matches", f"{exact_match_count} / {len(results)}")
        col3.metric("Line 1 Avg Accuracy", f"{total_l1_acc / len(results):.1f}%")
        col4.metric("Line 2 Avg Accuracy", f"{total_l2_acc / len(results):.1f}%")

        # Detailed Summary Table
        summary_rows = []
        for r in results:
            summary_rows.append(
                {
                    "Image Name": r["filename"],
                    "Status": "Exact Match ✅" if r["is_exact"] else ("Partial Match ⚠️" if r["det_l1"] else "Failed ❌"),
                    "Line 1 Accuracy": f"{r['comp_l1']['accuracy']:.1f}%",
                    "Line 2 Accuracy": f"{r['comp_l2']['accuracy']:.1f}%",
                    "Line 1 Match": "✅" if r["comp_l1"]["is_exact_match"] else "❌",
                    "Line 2 Match": "✅" if r["comp_l2"]["is_exact_match"] else "❌",
                    "OCR Confidence": f"{r['output'].confidence * 100:.1f}%",
                }
            )

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Detailed breakdown accordion
        st.subheader("🔍 Detailed Image-by-Image Comparison")
        for r in results:
            status_icon = "✅" if r["is_exact"] else "❌"
            with st.expander(f"{status_icon} {r['filename']} — L1: {r['comp_l1']['accuracy']:.1f}%, L2: {r['comp_l2']['accuracy']:.1f}%"):
                render_comparison_details(
                    r["filename"],
                    r["act_l1"],
                    r["act_l2"],
                    r["det_l1"],
                    r["det_l2"],
                    r["comp_l1"],
                    r["comp_l2"],
                    r["output"],
                )


def render_single_upload_section(detector: Any, gt_data: Dict[str, Dict[str, str]]) -> None:
    uploaded_file = st.file_uploader(
        "Upload Passport Image to Compare",
        type=["png", "jpg", "jpeg", "bmp"],
        key="single_cmp_uploader",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        with col2:
            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / uploaded_file.name
                tmp_path.write_bytes(uploaded_file.getvalue())

                with st.spinner("Detecting MRZ..."):
                    output = detector.detect_from_passport(tmp_path)

        gt_info = gt_data.get(uploaded_file.name.lower(), {})
        if not gt_info:
            st.info(
                f"Note: Ground truth not automatically matched for filename '{uploaded_file.name}'. "
                "You can manually enter the expected MRZ lines below if needed."
            )

        act_l1 = st.text_input("Actual MRZ Line 1 (Ground Truth)", value=gt_info.get("line1", ""))
        act_l2 = st.text_input("Actual MRZ Line 2 (Ground Truth)", value=gt_info.get("line2", ""))

        det_l1 = output.mrz_line1 or ""
        det_l2 = output.mrz_line2 or ""

        comp_l1 = compare_lines(act_l1, det_l1)
        comp_l2 = compare_lines(act_l2, det_l2)

        st.subheader("Comparison Result")
        render_comparison_details(
            uploaded_file.name,
            act_l1,
            act_l2,
            det_l1,
            det_l2,
            comp_l1,
            comp_l2,
            output,
        )


def render_batch_upload_section(detector: Any, gt_data: Dict[str, Dict[str, str]]) -> None:
    uploaded_files = st.file_uploader(
        "Upload Batch Passport Images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key="batch_cmp_uploader",
    )

    if uploaded_files:
        if st.button("Compare Batch", type="primary"):
            results = []
            with TemporaryDirectory() as tmp_dir:
                progress_bar = st.progress(0)
                for idx, file in enumerate(uploaded_files):
                    tmp_path = Path(tmp_dir) / file.name
                    tmp_path.write_bytes(file.getvalue())
                    output = detector.detect_from_passport(tmp_path)

                    gt_info = gt_data.get(file.name.lower(), {})
                    act_l1 = gt_info.get("line1", "")
                    act_l2 = gt_info.get("line2", "")

                    det_l1 = output.mrz_line1 or ""
                    det_l2 = output.mrz_line2 or ""

                    comp_l1 = compare_lines(act_l1, det_l1)
                    comp_l2 = compare_lines(act_l2, det_l2)

                    is_exact = comp_l1["is_exact_match"] and comp_l2["is_exact_match"]

                    results.append(
                        {
                            "filename": file.name,
                            "is_exact": is_exact,
                            "act_l1": act_l1,
                            "act_l2": act_l2,
                            "det_l1": det_l1,
                            "det_l2": det_l2,
                            "comp_l1": comp_l1,
                            "comp_l2": comp_l2,
                            "output": output,
                        }
                    )
                    progress_bar.progress((idx + 1) / len(uploaded_files))

            # Display Batch Summary
            st.subheader("Batch Comparison Results")
            summary_df = pd.DataFrame(
                [
                    {
                        "Filename": r["filename"],
                        "Status": "Match ✅" if r["is_exact"] else "Mismatch ❌",
                        "Line 1 Acc": f"{r['comp_l1']['accuracy']:.1f}%",
                        "Line 2 Acc": f"{r['comp_l2']['accuracy']:.1f}%",
                        "Detected L1": r["det_l1"],
                        "Actual L1": r["act_l1"],
                        "Detected L2": r["det_l2"],
                        "Actual L2": r["act_l2"],
                    }
                    for r in results
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            for r in results:
                with st.expander(f"{'✅' if r['is_exact'] else '❌'} {r['filename']}"):
                    render_comparison_details(
                        r["filename"],
                        r["act_l1"],
                        r["act_l2"],
                        r["det_l1"],
                        r["det_l2"],
                        r["comp_l1"],
                        r["comp_l2"],
                        r["output"],
                    )


def render_comparison_details(
    filename: str,
    act_l1: str,
    act_l2: str,
    det_l1: str,
    det_l2: str,
    comp_l1: Dict[str, Any],
    comp_l2: Dict[str, Any],
    output: Any,
) -> None:
    st.markdown("##### Line 1 Comparison")
    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.markdown(f"**Actual:** &nbsp; `{act_l1}`")
        st.markdown(f"**Detected:** `{det_l1}`")
        st.markdown(
            f"**Actual Diff:** &nbsp; {comp_l1['diff_actual_html']}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Detected Diff:** {comp_l1['diff_detected_html']}",
            unsafe_allow_html=True,
        )
    with col_b:
        st.metric("Line 1 Acc", f"{comp_l1['accuracy']:.1f}%")

    st.markdown("---")
    st.markdown("##### Line 2 Comparison")
    col_c, col_d = st.columns([4, 1])
    with col_c:
        st.markdown(f"**Actual:** &nbsp; `{act_l2}`")
        st.markdown(f"**Detected:** `{det_l2}`")
        st.markdown(
            f"**Actual Diff:** &nbsp; {comp_l2['diff_actual_html']}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Detected Diff:** {comp_l2['diff_detected_html']}",
            unsafe_allow_html=True,
        )
    with col_d:
        st.metric("Line 2 Acc", f"{comp_l2['accuracy']:.1f}%")

    if output.mrz_result:
        res = output.mrz_result
        st.markdown("##### Parsed Data Fields")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Passport #", res.passport_number or "N/A")
        f2.metric("Surname", res.surname or "N/A")
        f3.metric("Given Names", res.given_names or "N/A")
        f4.metric("Nationality", res.nationality or "N/A")
