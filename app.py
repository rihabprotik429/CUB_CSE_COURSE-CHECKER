import re
from io import BytesIO
import pandas as pd
import streamlit as st

# ---------------- Page + Headings ----------------
st.set_page_config(page_title="Course Checker", page_icon="✅", layout="wide")

# Edit these 3 lines if you need to tweak the text
st.title("CANADIAN UNIVERSITY OF BANGLADESH")
st.header("Department of CSE")
st.subheader("Course checker")

st.write("- Completion = a course code appears in the student data (grades ignored)")
st.write("- Supports CORE (all required), OPT1 (choose N), OPT2 (choose K pairs: lecture+lab)")
st.write("- Paste text copied from your university PDF, and/or upload a CSV/Excel of completed courses")

# ---------------- Helpers ----------------
def read_any(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def normalize_codes(s: pd.Series) -> pd.Series:
    # Uppercase; remove spaces and common dash variants (ASCII -, en/em dash, minus, NBSP)
    return (
        s.astype(str)
         .str.upper()
         .str.replace('[\\s\\-\\u2010-\\u2015\\u2212\\u00A0]', '', regex=True)
    )

def guess_col(cols, candidates):
    norm = [re.sub(r'[^a-z0-9_]', '', c.lower().strip().replace(" ", "_")) for c in cols]
    for idx, n in enumerate(norm):
        if n in candidates:
            return cols[idx]
    return cols[0] if cols else None

def extract_prefixes_from_catalog(series_like):
    # Build allowed course prefixes from catalog (e.g., CSE, EEE, MATH, ENG, PHY, BAN, ECO, ACT)
    prefixes = set()
    for val in series_like.dropna().astype(str):
        m = re.match(r'\s*([A-Za-z]{2,10})', val)
        if m:
            prefixes.add(m.group(1).upper())
    return prefixes

def parse_completed_from_text(text: str, allowed_prefixes, allowed_codes_norm):
    """
    Parse course codes from a pasted text blob.
    - Matches: CSE-1101, CSE 1101, CSE1101
    - Handles glued credit like MATH-11633 -> MATH-1163 (keeps first 4 digits)
    - Finds codes even if jammed like ...-TCSE-3303 (still finds CSE-3303)
    Returns DataFrame with 'course_code'.
    """
    if not text:
        return pd.DataFrame({"course_code": []})

    prefixes_sorted = sorted({p.upper() for p in allowed_prefixes}, key=len, reverse=True)
    if not prefixes_sorted:
        return pd.DataFrame({"course_code": []})
    P = "(?:" + "|".join(map(re.escape, prefixes_sorted)) + ")"

    DASH_WS = r'[\s\-\u2010-\u2015\u2212\u00A0]*'
    pat_strict = re.compile(rf'({P}){DASH_WS}(\d{{4}}[A-Za-z]?)', re.IGNORECASE)
    pat_joined = re.compile(rf'({P}){DASH_WS}(\d{{5}})', re.IGNORECASE)

    found = []
    def add_candidate(pref: str, num: str):
        pref = pref.upper().strip()
        num_up = num.upper()
        digits4 = num_up[:4]
        suffix = num_up[4] if len(num_up) >= 5 and num_up[4].isalpha() else ""
        norm = f"{pref}{digits4}{suffix}"
        if allowed_codes_norm and norm not in allowed_codes_norm:
            return
        found.append(f"{pref}-{digits4}{suffix}")

    for pref, num in pat_strict.findall(text):
        add_candidate(pref, num)
    for pref, num in pat_joined.findall(text):
        add_candidate(pref, num)

    # De-dup preserving original order of appearance
    seen = set()
    ordered = []
    for code in found:
        if code not in seen:
            seen.add(code)
            ordered.append(code)

    return pd.DataFrame({"course_code": ordered})

def pairs_table(df, cat_code_col, cat_title_col):
    rows = []
    for _, r in df.iterrows():
        for code, title in zip(r["codes"], r["titles"]):
            rows.append({
                "Pair": r["pair_id"],
                "Course Code": code,
                "Course Title": title,
                "Pair Credits": r["pair_credits"],
                "Complete": r["complete"]
            })
    return pd.DataFrame(rows)

def show_df(df, title, cat_code_col, cat_title_col):
    if df.empty:
        st.write(f"{title}: none")
        return pd.DataFrame()
    st.subheader(title)
    cols = [cat_code_col] + ([] if cat_title_col is None else [cat_title_col]) + ["__credits"]
    df_show = df[cols].copy()
    df_show.columns = ["Course Code"] + ([] if cat_title_col is None else ["Course Title"]) + ["Credits"]
    st.dataframe(df_show, use_container_width=True)
    return df_show

# ---------------- UI: inputs ----------------
col_left, col_right = st.columns([1,1])

with col_left:
    st.subheader("1) Upload Catalog (.xlsx/.csv)")
    cat_file = st.file_uploader("Catalog (with credits; group/pair optional)", type=["csv","xlsx","xls"])

with col_right:
    st.subheader("2) Provide Completed courses")
    st.write("Use the paste box and/or CSV/Excel (they will be combined).")
    paste_text = st.text_area(
        "Paste text copied from the university PDF here",
        height=220,
        placeholder="Example:\n22204001SUMMER-22 CSE-1101 3\n22204001SUMMER-22 MATH-11633\n..."
    )
    done_file = st.file_uploader("Completed (CSV/Excel) - optional", type=["csv","xlsx","xls"])

if cat_file:
    cat_raw = read_any(cat_file)
    st.write("Catalog preview:", cat_raw.head(5))

    st.subheader("3) Map catalog columns")
    cat_code_guess = guess_col(cat_raw.columns, {"course_code","code","courseid","course_id","course"})
    cat_cred_guess = guess_col(cat_raw.columns, {"credits","credit","cr","units","hours"})
    cat_title_guess = guess_col(cat_raw.columns, {"title","course_title","name","course_name"}) if len(cat_raw.columns) > 1 else None
    cat_group_guess = guess_col(cat_raw.columns, {"group","type","category"}) if len(cat_raw.columns) > 1 else None
    cat_pair_guess = guess_col(cat_raw.columns, {"pair","bundle","pair_id"}) if len(cat_raw.columns) > 1 else None

    cat_code_col = st.selectbox("Catalog: course code column", options=list(cat_raw.columns),
                                index=list(cat_raw.columns).index(cat_code_guess) if cat_code_guess in cat_raw.columns else 0)
    cat_cred_col = st.selectbox("Catalog: credits column", options=list(cat_raw.columns),
                                index=list(cat_raw.columns).index(cat_cred_guess) if cat_cred_guess in cat_raw.columns else 0)
    cat_title_col = st.selectbox("Catalog: title column (optional)", options=["-- none --"] + list(cat_raw.columns),
                                 index=0 if not cat_title_guess or cat_title_guess not in cat_raw.columns else 1 + list(cat_raw.columns).index(cat_title_guess))
    cat_group_col = st.selectbox("Catalog: group column (CORE/OPT1/OPT2) (optional)", options=["-- none --"] + list(cat_raw.columns),
                                 index=0 if not cat_group_guess or cat_group_guess not in cat_raw.columns else 1 + list(cat_raw.columns).index(cat_group_guess))
    cat_pair_col = st.selectbox("Catalog: OPT2 pair column (same label for lecture+lab) (optional)", options=["-- none --"] + list(cat_raw.columns),
                                index=0 if not cat_pair_guess or cat_pair_guess not in cat_raw.columns else 1 + list(cat_raw.columns).index(cat_pair_guess))

    st.subheader("4) Rules")
    use_groups = (cat_group_col != "-- none --")
    opt1_required = st.number_input("OPT1: number of courses required", min_value=0, max_value=10, value=1, step=1, disabled=not use_groups)
    opt2_pairs_required = st.number_input("OPT2: number of pairs required", min_value=0, max_value=10, value=2, step=1, disabled=not use_groups)
    st.caption("If you skip group/pair columns, everything is treated as CORE.")

    if st.button("Generate report"):
        # Prepare catalog
        keep_cols = [cat_code_col, cat_cred_col]
        title_col_real = None if cat_title_col == "-- none --" else cat_title_col
        if title_col_real: keep_cols.append(title_col_real)
        if use_groups: keep_cols.append(cat_group_col)
        if cat_pair_col != "-- none --": keep_cols.append(cat_pair_col)

        cat = cat_raw[keep_cols].copy()
        cat["__code_norm"] = normalize_codes(cat[cat_code_col])
        cat["__credits"] = pd.to_numeric(cat[cat_cred_col], errors="coerce").fillna(0)
        if use_groups:
            cat["__group"] = cat[cat_group_col].astype(str).str.upper().str.strip()
        else:
            cat["__group"] = "CORE"
        if cat_pair_col != "-- none --":
            cat["__pair"] = cat[cat_pair_col].astype(str).str.strip()
        else:
            cat["__pair"] = None
        cat = cat.sort_values(by=[cat_code_col]).drop_duplicates(subset="__code_norm", keep="first")

        # Warn if OPT2 exists but no pair labels
        if use_groups and "OPT2" in set(cat["__group"]) and cat_pair_col == "-- none --":
            st.warning("OPT2 rows found but no Pair column provided. Each OPT2 lecture+lab should share a pair label for accurate pairing.")

        # Allowed filters for parsing
        allowed_prefixes = extract_prefixes_from_catalog(cat[cat_code_col])
        allowed_codes_norm = set(cat["__code_norm"].tolist())

        # Gather completed codes from paste and optional CSV/Excel
        pass_codes = set()

        # From paste
        if paste_text.strip():
            parsed_df = parse_completed_from_text(paste_text, allowed_prefixes, allowed_codes_norm)
            st.write(f"Parsed {len(parsed_df)} course code(s) from pasted text.")
            if len(parsed_df):
                parsed_df["__code_norm"] = normalize_codes(parsed_df["course_code"])
                pass_codes |= set(parsed_df["__code_norm"].dropna().unique())

        # From CSV/Excel (optional)
        if done_file:
            done_raw = read_any(done_file)
            done_code_guess = guess_col(done_raw.columns, {"course_code","code","courseid","course_id","course"})
            done_code_col = st.selectbox("Completed: course code column (for CSV/Excel)", options=list(done_raw.columns),
                                         index=list(done_raw.columns).index(done_code_guess) if done_code_guess in done_raw.columns else 0)
            done = done_raw[[done_code_col]].copy()
            done["__code_norm"] = normalize_codes(done[done_code_col])
            pass_codes |= set(done["__code_norm"].dropna().unique())

        if not pass_codes:
            st.warning("No completed course codes found. Paste text and/or upload a CSV/Excel, then click Generate report.")
            st.stop()

        # Completed vs missing overall
        completed_all = cat[cat["__code_norm"].isin(pass_codes)].copy()
        missing_all = cat[~cat["__code_norm"].isin(pass_codes)].copy()

        # Extras: codes present in completed but not in catalog
        extra_codes = sorted(list(pass_codes - set(cat["__code_norm"])))
        extras = pd.DataFrame({"Course Code": extra_codes}) if extra_codes else pd.DataFrame({"Course Code": []})

        # Split by group
        core = cat[cat["__group"] == "CORE"].copy()
        opt1 = cat[cat["__group"] == "OPT1"].copy()
        opt2 = cat[cat["__group"] == "OPT2"].copy()

        # CORE
        core_completed = core[core["__code_norm"].isin(pass_codes)].copy()
        core_missing = core[~core["__code_norm"].isin(pass_codes)].copy()
        core_total_credits = float(core["__credits"].sum())
        core_earned_credits = float(core_completed["__credits"].sum())

        # OPT1: any N courses
        opt1_taken = opt1[opt1["__code_norm"].isin(pass_codes)].copy()
        opt1_taken_sorted = opt1_taken.sort_values("__credits", ascending=False)
        opt1_count_taken = len(opt1_taken_sorted)
        opt1_count_needed = max(0, opt1_required - opt1_count_taken)
        opt1_credited = float(opt1_taken_sorted["__credits"].head(opt1_required).sum()) if opt1_required > 0 else 0.0

        # OPT2: pairs (lecture+lab)
        opt2_pairs = []
        if not opt2.empty:
            pair_groups = opt2.groupby("__pair", dropna=False)
            for pid, g in pair_groups:
                codes = list(g["__code_norm"])
                titles = list(g[title_col_real]) if title_col_real else list(g[cat_code_col])
                credits = float(g["__credits"].sum())
                present = [c in pass_codes for c in codes]
                complete = all(present) and len(codes) > 0
                opt2_pairs.append({
                    "pair_id": pid if pd.notna(pid) else "(no pair label)",
                    "codes": codes,
                    "titles": titles,
                    "pair_credits": credits,
                    "complete": complete,
                    "completed_count": sum(present),
                    "required_count": len(codes),
                    "missing_codes": [c for c, p in zip(codes, present) if not p]
                })
        opt2_pairs_df = pd.DataFrame(opt2_pairs)

        opt2_pairs_completed = 0
        opt2_credited = 0.0
        opt2_pairs_needed = 0
        opt2_completed_list = pd.DataFrame()
        opt2_partial_list = pd.DataFrame()
        opt2_remaining_list = pd.DataFrame()

        if not opt2_pairs_df.empty:
            completed_pairs = opt2_pairs_df[opt2_pairs_df["complete"]].copy()
            partial_pairs = opt2_pairs_df[(~opt2_pairs_df["complete"]) & (opt2_pairs_df["completed_count"] > 0)].copy()
            remaining_pairs = opt2_pairs_df[opt2_pairs_df["completed_count"] == 0].copy()
            opt2_pairs_completed = len(completed_pairs)
            opt2_pairs_needed = max(0, opt2_pairs_required - opt2_pairs_completed)
            opt2_credited = float(completed_pairs["pair_credits"].sort_values(ascending=False).head(opt2_pairs_required).sum())

            opt2_completed_list = pairs_table(completed_pairs, cat_code_col, title_col_real)
            opt2_partial_list = pairs_table(partial_pairs, cat_code_col, title_col_real)
            opt2_remaining_list = pairs_table(remaining_pairs, cat_code_col, title_col_real)

        # Estimate required credits cap (for display)
        best_opt1_credit = float(opt1["__credits"].max()) if not opt1.empty and opt1_required > 0 else 0.0
        best_opt2_needed = 0.0
        if not opt2_pairs_df.empty and opt2_pairs_required > 0:
            best_opt2_needed = float(opt2_pairs_df["pair_credits"].sort_values(ascending=False).head(opt2_pairs_required).sum())
        program_required_credits_est = core_total_credits + best_opt1_credit + best_opt2_needed

        earned_within_requirement = core_earned_credits + opt1_credited + opt2_credited
        remaining_credits_est = max(0.0, program_required_credits_est - earned_within_requirement)

        # ---------------- Results (like the first version) ----------------
        st.subheader("Summary")
        st.write(f"CORE: {len(core_completed)}/{len(core)} completed | Credits: {core_earned_credits:.2f}/{core_total_credits:.2f}")
        if use_groups and not opt1.empty:
            st.write(f"OPT1: {opt1_count_taken}/{opt1_required} chosen | Credits counted: {opt1_credited:.2f}")
        if use_groups and not opt2.empty:
            st.write(f"OPT2: {opt2_pairs_completed}/{opt2_pairs_required} pairs completed | Credits counted: {opt2_credited:.2f}")
        st.write(f"Estimated program credits required: {program_required_credits_est:.2f}")
        st.write(f"Credits earned (within requirement cap): {earned_within_requirement:.2f} | Remaining (est): {remaining_credits_est:.2f}")

        comp_show = show_df(core_completed, "CORE - Completed", cat_code_col, title_col_real)
        miss_show = show_df(core_missing, "CORE - Missing", cat_code_col, title_col_real)

        if use_groups and not opt1.empty:
            st.subheader("OPT1")
            _ = show_df(opt1_taken, "OPT1 - Taken", cat_code_col, title_col_real)
            if opt1_count_needed > 0:
                st.write(f"Still need {opt1_count_needed} course(s) from OPT1.")
                opt1_options = opt1[~opt1["__code_norm"].isin(opt1_taken["__code_norm"])].copy()
                _ = show_df(opt1_options, "OPT1 - Available options", cat_code_col, title_col_real)

        if use_groups and not opt2.empty:
            st.subheader("OPT2 pairs")
            if not opt2_completed_list.empty:
                st.write("Completed pairs")
                st.dataframe(opt2_completed_list, use_container_width=True)
            if not opt2_partial_list.empty:
                st.write("Partially completed pairs (missing at least one course in the pair)")
                st.dataframe(opt2_partial_list, use_container_width=True)
            if not opt2_remaining_list.empty:
                st.write("Pairs with no courses taken yet")
                st.dataframe(opt2_remaining_list, use_container_width=True)
            if opt2_pairs_needed > 0:
                st.write(f"Still need {opt2_pairs_needed} pair(s) from OPT2.")

        # ---------------- Excel export ----------------
        summary_df = pd.DataFrame([{
            "CORE courses": f"{len(core_completed)}/{len(core)}",
            "CORE credits earned": core_earned_credits,
            "CORE credits required": core_total_credits,
            "OPT1 chosen": f"{len(opt1_taken)}/{opt1_required}" if use_groups else "N/A",
            "OPT1 credits counted": opt1_credited if use_groups else 0.0,
            "OPT2 pairs completed": f"{opt2_pairs_completed}/{opt2_pairs_required}" if use_groups else "N/A",
            "OPT2 credits counted": opt2_credited if use_groups else 0.0,
            "Estimated program credits required": program_required_credits_est,
            "Credits earned (within requirement cap)": earned_within_requirement,
            "Estimated credits remaining": remaining_credits_est
        }])

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            if not comp_show.empty: comp_show.to_excel(writer, index=False, sheet_name="CORE_Completed")
            if not miss_show.empty: miss_show.to_excel(writer, index=False, sheet_name="CORE_Missing")
            if use_groups and not opt1.empty:
                opt1_taken_show = opt1[[cat_code_col] + ([] if title_col_real is None else [title_col_real]) + ["__credits"]]
                opt1_taken_show.columns = ["Course Code"] + ([] if title_col_real is None else ["Course Title"]) + ["Credits"]
                opt1_taken_show.to_excel(writer, index=False, sheet_name="OPT1_Taken")
                opt1_all_show = opt1[[cat_code_col] + ([] if title_col_real is None else [title_col_real]) + ["__credits"]]
                opt1_all_show.columns = ["Course Code"] + ([] if title_col_real is None else ["Course Title"]) + ["Credits"]
                opt1_all_show.to_excel(writer, index=False, sheet_name="OPT1_All")
            if use_groups and not opt2.empty:
                if not opt2_completed_list.empty:
                    opt2_completed_list.to_excel(writer, index=False, sheet_name="OPT2_CompletedPairs")
                if not opt2_partial_list.empty:
                    opt2_partial_list.to_excel(writer, index=False, sheet_name="OPT2_PartialPairs")
                if not opt2_remaining_list.empty:
                    opt2_remaining_list.to_excel(writer, index=False, sheet_name="OPT2_RemainingPairs")
            if not extras.empty:
                extras.to_excel(writer, index=False, sheet_name="Extras")

        st.download_button("Download report (Excel)", data=output.getvalue(),
                           file_name="course_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload Catalog first, then paste the text (or upload Completed CSV/Excel) and click Generate report.")