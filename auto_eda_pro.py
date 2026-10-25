import os
import re
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# =========================================================================
# GLOBAL CONFIGURATION & DOMAIN DICTIONARIES
# =========================================================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

YOUTUBE_CATEGORIES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events",
    "20": "Gaming", "22": "People & Blogs", "23": "Comedy",
    "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
    "27": "Education", "28": "Science & Technology",
    "29": "Nonprofits & Activism", "43": "Shows",
}

DOMAIN_CAVEATS = {
    "Social Media / YouTube Analytics": "Dataset includes ONLY trending videos — findings describe what trending looks like, not what causes trending.",
    "Academic / Student Performance Analytics": "Correlation ≠ causation — attendance and CGPA may share latent confounders.",
    "E-Commerce / Finance / Retail Analytics": "Revenue/sales data is cumulative and time-dependent — do not compare across periods without normalizing for volume.",
    "HR / People Analytics": "Observational workforce data reflects existing policies — interventions may alter baseline behavior.",
    "Healthcare / Clinical Analytics": "Clinical records reflect patient presentation bias — unrecorded comorbidities may confound relationships.",
    "General Tabular Analytics": "Findings are descriptive and correlational based on observed records; verify via controlled experimentation.",
}

DOMAIN_RECOMMENDATIONS = {
    "Social Media / YouTube Analytics": [
        "Prioritise the dominant category — it owns the largest share of trending slots.",
        "Optimise for engagement rate, not raw views — high likes/views predicts longevity.",
        "Publish during high-volume months (see time-series chart).",
        "Study videos with highest 'days_trending' — durable wins live there.",
        "Avoid mean-based comparisons — a few viral outliers distort the mean.",
    ],
    "Academic / Student Performance Analytics": [
        "Focus support programs on subjects or grade bands with highest failure densities.",
        "Investigate students with low attendance AND low performance metrics simultaneously.",
        "Use monotonic ID columns strictly for database joins, never for analytical modeling.",
        "Monitor grade distributions for skewness before applying parametric tests.",
    ],
    "E-Commerce / Finance / Retail Analytics": [
        "Segment customer cohorts by recency and monetary value rather than aggregate averages.",
        "Investigate IQR outlier transactions for fraud, bulk wholesale, or data entry errors.",
        "Normalize seasonal sales trends before running predictive forecasting models.",
    ],
    "HR / People Analytics": [
        "Examine tenure brackets where attrition spikes to target retention initiatives.",
        "Evaluate compensation equity across departments controlling for experience levels.",
    ],
    "Healthcare / Clinical Analytics": [
        "Prioritise early-detection features (BMI, glucose, vitals) over late-stage outcomes.",
        "Check for demographic confounding before establishing clinical correlations.",
    ],
    "General Tabular Analytics": [
        "Address columns exceeding 50% missingness before building predictive models.",
        "Transform highly skewed numeric features using log or Box-Cox scaling.",
        "Cross-reference IQR outlier records with domain experts prior to removal.",
    ],
}

DERIVED_COLS = {
    "engagement_rate", "comment_rate", "like_dislike_ratio", "days_trending",
    "title_length", "tag_count", "video_age_days"
}

# =========================================================================
# UNIVERSAL I/O & FORMATTING HELPERS
# =========================================================================
def load_any_dataset(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    base = os.path.basename(file_path)

    try:
        if ext in [".json", ".jsonl", ".ndjson"]:
            df = None
            for lines_mode in ([True, False] if ext != ".json" else [False, True]):
                try:
                    df = pd.read_json(file_path, lines=lines_mode)
                    break
                except Exception:
                    continue

            if df is None or (len(df.columns) <= 2 and len(df) > 0 and any(isinstance(v, (list, dict)) for v in df.iloc[0].values)):
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_data = json.load(f)

                if isinstance(raw_data, list):
                    df = pd.json_normalize(raw_data, sep="_")
                elif isinstance(raw_data, dict):
                    list_keys = [k for k, v in raw_data.items() if isinstance(v, list) and len(v) > 0]
                    if list_keys:
                        best_key = max(list_keys, key=lambda k: len(raw_data[k]))
                        df = pd.json_normalize(raw_data[best_key], sep="_")
                    else:
                        df = pd.json_normalize(raw_data, sep="_")

            for col in list(df.columns):
                sample_vals = df[col].dropna().head(20)
                if any(isinstance(x, list) for x in sample_vals):
                    df[col] = df[col].apply(lambda x: "|".join(map(str, x)) if isinstance(x, list) else x)
                elif any(isinstance(x, dict) for x in sample_vals):
                    df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)

            print(f"✅ Loaded and flattened JSON '{base}' ({df.shape[0]:,} rows × {df.shape[1]} cols)")
            return df

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            print(f"✅ Loaded Excel file '{base}' ({df.shape[0]:,} rows × {df.shape[1]} cols)")
            return df

        elif ext == ".parquet":
            try:
                df = pd.read_parquet(file_path)
                print(f"✅ Loaded Parquet file '{base}' ({df.shape[0]:,} rows × {df.shape[1]} cols)")
                return df
            except ImportError:
                print("❌ Parquet support requires 'pyarrow' or 'fastparquet'. Run: pip install pyarrow")
                return None

        else:
            sep = "\t" if ext == ".tsv" else ","
            encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=enc, low_memory=False, on_bad_lines="warn")
                    print(f"✅ Loaded tabular file '{base}' (Encoding: {enc})")
                    return df
                except UnicodeDecodeError:
                    continue
            print("❌ Failed to decode file with standard encodings.")
            return None

    except Exception as e:
        print(f"❌ Error loading '{base}': {e}")
        return None


def standardize_column_names(columns):
    clean_cols, seen = [], {}
    for col in columns:
        c = str(col).strip()
        c = re.sub(r"[^\w\s]", "_", c)
        c = re.sub(r"\s+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_").lower()
        if not c:
            c = "unnamed_col"
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        clean_cols.append(c)
    return clean_cols


def ask_yes_no(prompt_text, default="n"):
    choice = input(prompt_text).strip().lower()
    if not choice:
        choice = default
    return choice.startswith("y")


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|\s]+', "_", str(name))[:50]


def smart_parse_dates(series):
    sample = series.dropna().astype(str).head(50)
    if len(sample) == 0:
        return pd.Series(dtype="datetime64[ns]")
    if sample.str.match(r"^\d{2}\.\d{2}\.\d{2}$").mean() > 0.8:
        return pd.to_datetime(series, format="%y.%d.%m", errors="coerce")
    
    if pd.__version__ >= "2.0.0":
        return pd.to_datetime(series, errors="coerce", format="mixed")
    else:
        return pd.to_datetime(series, errors="coerce")


# =========================================================================
# CONTEXT ANALYZER & SMART COLUMN CLASSIFIER
# =========================================================================
def analyze_dataset_context(df):
    cols_set = set(df.columns)
    all_words = " ".join(df.columns).lower()

    if {"video_id", "views"}.issubset(cols_set):
        domain = "Social Media / YouTube Analytics"
    elif any(k in all_words for k in ["marks", "grade", "cgpa", "sgpa", "exam", "student", "attendance", "roll"]):
        domain = "Academic / Student Performance Analytics"
    elif any(k in all_words for k in ["order", "sales", "revenue", "price", "customer", "product", "invoice", "expense"]):
        domain = "E-Commerce / Finance / Retail Analytics"
    elif any(k in all_words for k in ["salary", "employee", "department", "attrition", "tenure", "designation"]):
        domain = "HR / People Analytics"
    elif any(k in all_words for k in ["patient", "blood", "diagnosis", "glucose", "bmi", "hospital", "clinical"]):
        domain = "Healthcare / Clinical Analytics"
    else:
        domain = "General Tabular Analytics"

    id_keywords = {
        "id", "uuid", "roll", "roll_no", "serial", "sr_no", "index",
        "zip", "zipcode", "pincode", "phone", "mobile", "contact", "code", "category_id"
    }
    smart_num_cols = []
    excluded_id_cols = []

    for col in df.select_dtypes(include="number").columns:
        if pd.api.types.is_bool_dtype(df[col]) or df[col].nunique() <= 1:
            continue
        col_lower = col.lower()
        is_id_name = any(
            col_lower == kw or col_lower.startswith(f"{kw}_") or col_lower.endswith(f"_{kw}")
            for kw in id_keywords
        )
        is_monotonic_id = (
            pd.api.types.is_integer_dtype(df[col])
            and df[col].nunique() >= 0.95 * len(df)
            and len(df) > 30
        )
        if is_id_name or is_monotonic_id:
            excluded_id_cols.append(col)
        else:
            smart_num_cols.append(col)

    priority_keywords = [
        "sales", "revenue", "amount", "total", "price", "profit", "cost", "expense",
        "marks", "score", "cgpa", "grade", "percentage",
        "salary", "income", "views", "likes", "engagement_rate"
    ]

    def metric_priority(col_name):
        for idx, kw in enumerate(priority_keywords):
            if kw in col_name.lower():
                return idx
        return 999

    smart_num_cols.sort(key=metric_priority)
    primary_metric = smart_num_cols[0] if smart_num_cols else None

    month_cols = {c for c in df.columns if c.endswith("_month")}
    text_noise_keywords = {"id", "uuid", "url", "link", "description", "title", "name", "email", "address", "phone", "tags"}

    smart_cat_cols = []
    for c in df.select_dtypes(include=["object", "category", "bool", "string"]).columns:
        if c in month_cols:
            continue
        if any(kw in c.lower() for kw in text_noise_keywords):
            excluded_id_cols.append(c)
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= min(25, max(10, int(len(df) * 0.4))):
            smart_cat_cols.append(c)

    if "category_name" in smart_cat_cols:
        smart_cat_cols.remove("category_name")
        smart_cat_cols.insert(0, "category_name")

    return {
        "domain": domain,
        "num_cols": smart_num_cols,
        "primary_metric": primary_metric,
        "cat_cols": smart_cat_cols,
        "excluded_cols": excluded_id_cols,
        "month_cols": sorted(list(month_cols)),
    }


# =========================================================================
# CORE CLEANING PIPELINE
# =========================================================================
def clean_and_organize_dataset(df_raw, output_dir, drop_missing_threshold=0.70, auto_mode=False):
    df = df_raw.copy()
    cleaning_log = []
    initial_rows, initial_cols = df.shape
    duplicates_removed_count = 0

    old_cols = list(df.columns)
    df.columns = standardize_column_names(df.columns)
    if old_cols != list(df.columns):
        cleaning_log.append("Standardized column headers into SQL-ready snake_case.")

    hidden_nulls = {
        "", "nan", "null", "none", "n/a", "na", "?", "-", "--",
        "undefined", "nil", "[none]", "missing"
    }
    obj_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in obj_cols:
        df[col] = df[col].where(df[col].notnull(), np.nan)
        mask = df[col].notnull()
        df.loc[mask, col] = (
            df.loc[mask, col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )
        # FIX: Added isinstance(x, str) check
        df[col] = df[col].apply(
            lambda x: np.nan if isinstance(x, str) and x.lower() in hidden_nulls else x
        )

    for col in obj_cols:
        non_null_count = df[col].notnull().sum()
        if non_null_count == 0:
            continue

        cleaned_num = df[col].dropna().astype(str).str.replace(r"[$,₹€£,%]", "", regex=True).str.strip()
        converted_num = pd.to_numeric(cleaned_num, errors="coerce")
        if converted_num.notnull().sum() >= 0.85 * non_null_count:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,₹€£,%]", "", regex=True).str.strip(),
                errors="coerce",
            )
            cleaning_log.append(f"Converted formatted text column '{col}' -> numeric.")
            continue

        date_keywords = ["date", "time", "timestamp", "dt", "dob", "created", "updated", "joined"]
        if any(k in col for k in date_keywords):
            converted_dt = smart_parse_dates(df[col])
            if converted_dt.notnull().sum() >= 0.80 * non_null_count:
                df[col] = converted_dt
                cleaning_log.append(f"Parsed '{col}' as standardized datetime.")
                continue

        unique_raw = df[col].dropna().nunique()
        unique_lower = df[col].dropna().astype(str).str.lower().nunique()
        if unique_raw > unique_lower and unique_lower <= 100:
            df.loc[df[col].notnull(), col] = df.loc[df[col].notnull(), col].astype(str).str.title()
            cleaning_log.append(f"Unified text casing in '{col}' ({unique_raw} variants merged).")

    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
        cleaning_log.append(f"Dropped {len(empty_cols)} 100%-empty column(s): {empty_cols}")

    empty_rows = int(df.isnull().all(axis=1).sum())
    if empty_rows > 0:
        df.dropna(how="all", inplace=True)
        cleaning_log.append(f"Dropped {empty_rows:,} 100%-empty row(s).")

    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        print("\n" + "-" * 60)
        print(f"⚠️  CHECKPOINT 1: Found {dup_count:,} exact duplicate row(s) across all columns.")
        print(df[dup_mask].head(3).to_string(index=False))
        remove_dups = True if auto_mode else ask_yes_no(
            f"\n👉 Remove {dup_count:,} duplicate rows? (Backed up to quarantine_duplicates.csv) [y/N]: ",
            default="n",
        )
        if remove_dups:
            quarantine_path = os.path.join(output_dir, "quarantine_duplicates.csv")
            df[dup_mask].to_csv(quarantine_path, index=False)
            df.drop_duplicates(keep="first", inplace=True)
            duplicates_removed_count = dup_count
            cleaning_log.append(f"Removed {dup_count:,} duplicate row(s) (quarantined).")
            print(f"✅ Quarantined duplicates to: {quarantine_path}")
        else:
            cleaning_log.append(f"Retained {dup_count:,} duplicate row(s) per user choice.")

    high_missing_cols = {
        col: (df[col].isnull().sum() / len(df)) * 100
        for col in df.columns
        if (df[col].isnull().sum() / len(df)) > drop_missing_threshold
    }
    if high_missing_cols:
        print("\n" + "-" * 60)
        print(f"⚠️  CHECKPOINT 2: {len(high_missing_cols)} column(s) exceed {int(drop_missing_threshold*100)}% missing:")
        for c, pct in high_missing_cols.items():
            print(f"   • {c}: {pct:.1f}% missing")
        drop_cols = True if auto_mode else ask_yes_no(
            "\n👉 Drop these mostly-empty columns? [y/N]: ", default="n"
        )
        if drop_cols:
            cols_to_drop = list(high_missing_cols.keys())
            df.drop(columns=cols_to_drop, inplace=True)
            cleaning_log.append(f"Dropped high-missing columns: {cols_to_drop}")
        else:
            cleaning_log.append(f"Retained high-missing columns: {list(high_missing_cols.keys())}")

    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    if missing_cols:
        print("\n" + "-" * 60)
        print(f"⚠️  CHECKPOINT 3: {len(missing_cols)} column(s) contain missing values (NaN).")
        for c in missing_cols[:10]:
            cnt = int(df[c].isnull().sum())
            pct = (cnt / len(df)) * 100
            print(f"   • {c} ({df[c].dtype}): {cnt:,} missing ({pct:.1f}%)")

        if auto_mode:
            impute_choice = "2"
        else:
            print("\nHow would you like to handle missing values?")
            print("  [1] Keep NaN untouched (Best before ML train/test split)")
            print("  [2] Safe Fill: Median for numerics + 'Unknown' for text (Recommended)")
            print("  [3] Full Impute: Median for numerics + Mode for text")
            impute_choice = input("👉 Enter choice [1/2/3] (default = 1): ").strip() or "1"

        if impute_choice == "1":
            cleaning_log.append("Left missing values (NaN) untouched per user selection.")
        else:
            for col in missing_cols:
                missing_cnt = int(df[col].isnull().sum())
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    cleaning_log.append(f"Filled {missing_cnt:,} NaN in '{col}' with median ({median_val:.2f}).")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    cleaning_log.append(f"Left {missing_cnt:,} NaT datetime in '{col}' untouched.")
                else:
                    if impute_choice == "3":
                        mode_series = df[col].mode(dropna=True)
                        fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                        method_label = f"mode ('{fill_val}')"
                    else:
                        fill_val, method_label = "Unknown", "'Unknown'"
                    df[col] = df[col].fillna(fill_val)
                    cleaning_log.append(f"Filled {missing_cnt:,} NaN in '{col}' with {method_label}.")

    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    num_cols_ = df.select_dtypes(include="number").columns.tolist()
    other_cols = [c for c in df.columns if c not in dt_cols and c not in num_cols_]
    organized_cols = dt_cols + other_cols + num_cols_
    if list(df.columns) != organized_cols:
        df = df[organized_cols]
        cleaning_log.append("Reorganized columns: [DateTime -> Categorical/Text -> Numeric].")

    if dt_cols:
        df.sort_values(by=dt_cols[0], ascending=True, inplace=True, na_position="last")
        cleaning_log.append(f"Sorted rows chronologically by '{dt_cols[0]}'.")
    elif len(df.columns) > 0:
        df.sort_values(by=df.columns[0], ascending=True, inplace=True, na_position="last")

    df.reset_index(drop=True, inplace=True)
    final_rows, final_cols = df.shape
    cleaning_log.insert(
        0,
        f"Shape: ({initial_rows:,} rows, {initial_cols} cols) -> ({final_rows:,} rows, {final_cols} cols).",
    )

    return df, cleaning_log, duplicates_removed_count


# =========================================================================
# ADAPTIVE FEATURE ENGINEERING
# =========================================================================
def engineer_adaptive_features(df, cleaning_log):
    df = df.copy()
    is_youtube = {"video_id", "views"}.issubset(set(df.columns))

    if is_youtube:
        if "category_id" in df.columns:
            cat_str = df["category_id"].astype(str).str.replace(r"\.0$", "", regex=True)
            df["category_name"] = cat_str.map(YOUTUBE_CATEGORIES).fillna("Other")
            cleaning_log.append("[Domain Plugin] Mapped 'category_id' -> 'category_name'.")

        views_safe = df["views"].replace(0, np.nan)
        if "likes" in df.columns:
            df["engagement_rate"] = (df["likes"] / views_safe).fillna(0)
        if "comment_count" in df.columns:
            df["comment_rate"] = (df["comment_count"] / views_safe).fillna(0)
        df["days_trending"] = df.groupby("video_id")["video_id"].transform("size")
        
        if "title" in df.columns:
            df["title_length"] = df["title"].astype(str).str.len()
        if "tags" in df.columns:
            # FIX: Added not isinstance(x, str) check
            df["tag_count"] = df["tags"].apply(
                lambda x: 0 if not isinstance(x, str) or x.strip().lower() in ("", "nan", "[none]") else x.count("|") + 1
            )
        cleaning_log.append("[Domain Plugin] Engineered engagement_rate, comment_rate, days_trending, title_length, tag_count.")

    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if dt_cols:
        primary_dt = dt_cols[0]
        df[f"{primary_dt}_month"] = df[primary_dt].dt.to_period("M").astype(str)
        df[f"{primary_dt}_dayofweek"] = df[primary_dt].dt.day_name()
        cleaning_log.append(f"[Context Engine] Extracted '{primary_dt}_month' and '{primary_dt}_dayofweek'.")

    return df, cleaning_log


# =========================================================================
# OUTLIER DETECTION, QUALITY SCORECARD & CONTEXT INSIGHTS
# =========================================================================
def detect_outliers_iqr(df, num_cols):
    outlier_summary = {}
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0 or pd.api.types.is_bool_dtype(series):
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_summary[col] = 0
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_summary[col] = int(((series < lower) | (series > upper)).sum())
    return outlier_summary


def compute_quality_score(df_raw, raw_duplicates):
    total_cells = max(df_raw.shape[0] * df_raw.shape[1], 1)
    missing_cells = int(df_raw.isnull().sum().sum())
    completeness = 1 - (missing_cells / total_cells)
    uniqueness = 1 - (raw_duplicates / max(len(df_raw), 1))
    clean_cols_ratio = 1 - (df_raw.isnull().sum() > 0).mean() if len(df_raw.columns) > 0 else 1.0

    score = completeness * 50 + uniqueness * 30 + clean_cols_ratio * 20
    return {
        "completeness_pct": round(completeness * 100, 1),
        "uniqueness_pct": round(uniqueness * 100, 1),
        "source_clean_columns_pct": round(clean_cols_ratio * 100, 1),
        "overall_score": round(score, 1),
    }


def get_top_correlation_pair(df, num_cols):
    """Finds strongest correlation, excluding derived columns to avoid tautologies."""
    corr_candidates = [c for c in num_cols if c not in DERIVED_COLS]
    if len(corr_candidates) < 2:
        corr_candidates = num_cols

    if len(corr_candidates) < 2:
        return None

    corr_matrix = df[corr_candidates].corr()
    abs_corr = corr_matrix.abs().copy()
    
    # FIX: Make a writable copy to avoid ValueError
    vals = abs_corr.values.copy()
    np.fill_diagonal(vals, 0)
    abs_corr = pd.DataFrame(vals, index=abs_corr.index, columns=abs_corr.columns)
    
    if abs_corr.max().max() == 0:
        return None
        
    c1, c2 = abs_corr.stack().idxmax()
    return c1, c2, float(corr_matrix.loc[c1, c2])


def generate_context_insights(df, context, outlier_counts, top_corr):
    insights = [f"Detected Dataset Context: {context['domain']}."]
    num_cols = context["num_cols"]
    cat_cols = context["cat_cols"]
    primary_metric = context["primary_metric"]

    if context["excluded_cols"]:
        insights.append(
            f"Automatically excluded {len(context['excluded_cols'])} ID/text-noise column(s) from statistical charts."
        )

    if top_corr is not None:
        c1, c2, r_val = top_corr
        strength = "Strong" if abs(r_val) >= 0.7 else ("Moderate" if abs(r_val) >= 0.35 else "Weak")
        insights.append(f"{strength} linear relationship between '{c1}' and '{c2}' (r = {r_val:.2f}).")

    if primary_metric:
        med_v = df[primary_metric].median()
        mean_v = df[primary_metric].mean()
        skew_v = df[primary_metric].dropna().skew()
        insights.append(
            f"Primary Metric '{primary_metric}': Mean = {mean_v:,.2f}, Median = {med_v:,.2f} (Skewness = {skew_v:.2f})."
        )

    if cat_cols and primary_metric:
        top_cat = cat_cols[0]
        grp_mean = df.groupby(top_cat)[primary_metric].mean()
        grp_count = df.groupby(top_cat)[primary_metric].count()
        grp_score = grp_mean * np.log1p(grp_count)
        if len(grp_score) > 0:
            best_cat = grp_score.idxmax()
            insights.append(
                f"'{best_cat}' leads in '{primary_metric}' (mean: {grp_mean[best_cat]:,.2f} across {grp_count[best_cat]:,} records)."
            )

    if outlier_counts and sum(outlier_counts.values()) > 0:
        top_out = max(outlier_counts, key=outlier_counts.get)
        cnt = outlier_counts[top_out]
        insights.append(
            f"Outlier Hotspot: '{top_out}' contains {cnt:,} IQR outliers ({(cnt/len(df))*100:.1f}% of rows)."
        )

    return insights


# =========================================================================
# MAIN REPORT GENERATOR
# =========================================================================
def generate_eda_report(file_path, output_dir_arg=None, missing_thresh=0.70, auto_mode=False):
    file_path = file_path.strip().strip('"').strip("'")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df_raw = load_any_dataset(file_path)
    if df_raw is None or df_raw.empty:
        print("❌ Dataset empty or unloadable.")
        return

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir_arg:
        output_dir = os.path.join(output_dir_arg, f"EDA_Output_{sanitize_filename(base_name)}_{timestamp}")
    else:
        output_dir = f"EDA_Output_{sanitize_filename(base_name)}_{timestamp}"
        
    charts_dir = os.path.join(output_dir, "Charts")
    os.makedirs(charts_dir, exist_ok=True)

    raw_missing = df_raw.isnull().sum()
    raw_missing = raw_missing[raw_missing > 0]
    raw_duplicates = int(df_raw.duplicated().sum())

    print("\n🧹 Step 1/4: Running Glass-Box Cleaning & Organization...")
    df_clean, cleaning_log, dups_removed = clean_and_organize_dataset(
        df_raw, output_dir=output_dir, drop_missing_threshold=missing_thresh, auto_mode=auto_mode
    )

    print("🧠 Step 2/4: Analyzing Dataset Context & Engineering Features...")
    df_clean, cleaning_log = engineer_adaptive_features(df_clean, cleaning_log)
    context = analyze_dataset_context(df_clean)

    num_cols = context["num_cols"]
    cat_cols = context["cat_cols"]
    primary_metric = context["primary_metric"]
    month_cols = context["month_cols"]

    print(f"   • Detected Context : {context['domain']}")
    print(f"   • Primary Metric   : {primary_metric or 'None'}")
    print(f"   • Grouping Cols    : {cat_cols[:5] if cat_cols else 'None'}")

    cleaned_csv_path = os.path.join(output_dir, f"cleaned_{sanitize_filename(base_name)}.csv")
    df_clean.to_csv(cleaned_csv_path, index=False)
    print(f"✅ Saved cleaned CSV copy: {cleaned_csv_path}")

    outlier_counts = detect_outliers_iqr(df_clean, num_cols)
    quality = compute_quality_score(df_raw, raw_duplicates)
    top_corr = get_top_correlation_pair(df_clean, num_cols)
    insights = generate_context_insights(df_clean, context, outlier_counts, top_corr)

    metrics_payload = {
        "source_file": os.path.basename(file_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "detected_domain": context["domain"],
        "primary_metric": primary_metric,
        "analyzed_numeric_columns": num_cols,
        "analyzed_categorical_columns": cat_cols,
        "excluded_id_or_noise_columns": context["excluded_cols"],
        "raw_shape": [int(df_raw.shape[0]), int(df_raw.shape[1])],
        "clean_shape": [int(df_clean.shape[0]), int(df_clean.shape[1])],
        "quality_score": quality,
        "insights": insights,
        "cleaning_log": cleaning_log,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, default=str)

    pdf_path = os.path.join(output_dir, "EDA_Professional_Report.pdf")
    saved_charts = []

    print("🎨 Step 3/4: Building Signal-Driven Charts & PDF Report...")
    with PdfPages(pdf_path) as pdf:
        # Page 1: Cover with Domain Caveat
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        plt.axis("off")
        caveat_text = DOMAIN_CAVEATS.get(context["domain"], DOMAIN_CAVEATS["General Tabular Analytics"])
        cover = [
            "=" * 78,
            "UNIVERSAL CONTEXT-AWARE EDA & DATA ORGANIZATION REPORT",
            "=" * 78,
            "",
            f"Source File       : {os.path.basename(file_path)} (Original Untouched)",
            f"Detected Context  : {context['domain']}",
            f"Primary Metric    : {primary_metric or 'N/A'}",
            f"Generated On      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 78,
            "ANALYTICAL CAVEAT",
            "-" * 78,
            f"  {caveat_text}",
            "",
            "-" * 78,
            "DATASET HEALTH & STRUCTURE AT A GLANCE",
            "-" * 78,
            f"  Rows (Raw -> Clean)      : {df_raw.shape[0]:,} -> {df_clean.shape[0]:,}",
            f"  Columns (Raw -> Clean)   : {df_raw.shape[1]} -> {df_clean.shape[1]}",
            f"  Source Quality Score     : {quality['overall_score']} / 100",
            f"  Duplicates Removed       : {dups_removed:,} (of {raw_duplicates:,} detected)",
            f"  Genuine Numeric Metrics  : {len(num_cols)} (Excluded {len(context['excluded_cols'])} ID/noise cols)",
            f"  Total IQR Outliers       : {sum(outlier_counts.values()):,}",
            "",
            "-" * 78,
            "KEY CONTEXTUAL INSIGHTS",
            "-" * 78,
        ]
        for i, ins in enumerate(insights, 1):
            cover.append(f"  {i}. {ins}")
        plt.text(0.05, 0.95, "\n".join(cover), transform=fig.transFigure,
                 fontsize=10, verticalalignment="top", fontfamily="sans-serif")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Recommendations Page
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        plt.axis("off")
        recs = DOMAIN_RECOMMENDATIONS.get(context["domain"], DOMAIN_RECOMMENDATIONS["General Tabular Analytics"])
        recs_page = [
            "PRESCRIPTIVE RECOMMENDATIONS FOR STRATEGY & ACTIONS",
            "=" * 78,
            f"Tailored Domain Context: {context['domain']}",
            "-" * 78,
            "",
        ]
        for i, rec in enumerate(recs, 1):
            recs_page.append(f"  {i}. {rec}")
        plt.text(0.05, 0.95, "\n".join(recs_page), transform=fig.transFigure,
                 fontsize=11, verticalalignment="top", fontfamily="sans-serif")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Audit Log & Data Dictionary
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        plt.axis("off")
        page3 = [
            "CLEANING AUDIT LOG & DATA DICTIONARY",
            "=" * 82,
            f"{'COLUMN NAME':<28} {'DTYPE':<14} {'NON-NULL':>10} {'UNIQUE':>10} {'ASSIGNED ROLE':<16}",
            "-" * 82,
        ]
        for col in df_clean.columns[:30]:
            role = (
                "Primary Metric" if col == primary_metric
                else "Numeric Metric" if col in num_cols
                else "Category Group" if col in cat_cols
                else "ID / Text (Ignored)" if col in context["excluded_cols"]
                else "Date / Aux"
            )
            page3.append(
                f"{col[:27]:<28} {str(df_clean[col].dtype)[:13]:<14} "
                f"{df_clean[col].notnull().sum():>10,} {df_clean[col].nunique():>10,} {role:<16}"
            )
        plt.text(0.05, 0.95, "\n".join(page3), transform=fig.transFigure,
                 fontsize=8.5, verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Chart 1: Missing Value Density Map
        if df_raw.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            sns.heatmap(df_raw.isnull().T, cbar=False, cmap="viridis", ax=ax)
            ax.set_title("Missing Value Density Map (Raw Data Overview)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Record Index")
            ax.set_ylabel("Columns")
            plt.tight_layout()
            path = os.path.join(charts_dir, "missing_value_density.png")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved_charts.append(("Missing Value Density Map", "Charts/missing_value_density.png"))

        # Chart 2: Correlation Matrix
        if len(num_cols) >= 2:
            plot_num_cols = num_cols[:14]
            fig, ax = plt.subplots(figsize=(10, 7))
            corr_matrix = df_clean[plot_num_cols].corr()
            sns.heatmap(corr_matrix, annot=len(plot_num_cols) <= 12, cmap="coolwarm",
                        fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Correlation Matrix (Genuine Numeric Metrics Only)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(charts_dir, "correlation_heatmap.png")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved_charts.append(("Correlation Matrix", "Charts/correlation_heatmap.png"))

        # Chart 3: Top Correlation Scatter Plot
        if top_corr is not None and abs(top_corr[2]) >= 0.35:
            c1, c2, r_val = top_corr
            fig, ax = plt.subplots(figsize=(9, 5))
            sample_df = df_clean[[c1, c2]].dropna()
            if len(sample_df) > 2000:
                sample_df = sample_df.sample(2000, random_state=42)
            sns.regplot(
                data=sample_df, x=c1, y=c2, ax=ax,
                scatter_kws={"alpha": 0.5, "color": "royalblue"},
                line_kws={"color": "crimson", "linewidth": 2},
            )
            ax.set_title(f"Strongest Relationship: '{c1}' vs '{c2}' (r = {r_val:.2f})",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(charts_dir, "top_correlation_scatter.png")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved_charts.append((f"Relationship: {c1} vs {c2} (r={r_val:.2f})", "Charts/top_correlation_scatter.png"))

        # Chart 4: Distributions
        if len(num_cols) > 0:
            plot_cols = num_cols[:9]
            n_cols_plot = min(3, len(plot_cols))
            n_rows = int(np.ceil(len(plot_cols) / n_cols_plot))
            fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(5 * n_cols_plot, 4 * n_rows))
            axes = np.array(axes).flatten() if len(plot_cols) > 1 else [axes]
            for idx, col in enumerate(plot_cols):
                sns.histplot(df_clean[col].dropna(), kde=True, ax=axes[idx],
                             color="steelblue", edgecolor="black")
                skew_val = df_clean[col].dropna().skew()
                axes[idx].set_title(f"{col} (Skew: {skew_val:.2f})", fontsize=11, fontweight="bold")
            for idx in range(len(plot_cols), len(axes)):
                axes[idx].axis("off")
            fig.suptitle("Distributions of Key Numeric Metrics", fontsize=15, fontweight="bold", y=1.02)
            plt.tight_layout()
            path = os.path.join(charts_dir, "numeric_distributions.png")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved_charts.append(("Numeric Metric Distributions", "Charts/numeric_distributions.png"))

        # Chart 5: Outliers
        cols_with_outliers = [c for c in num_cols if outlier_counts.get(c, 0) > 0][:9]
        if cols_with_outliers:
            n_cols_plot = min(3, len(cols_with_outliers))
            n_rows = int(np.ceil(len(cols_with_outliers) / n_cols_plot))
            fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(5 * n_cols_plot, 3.5 * n_rows))
            axes = np.array(axes).flatten() if len(cols_with_outliers) > 1 else [axes]
            for idx, col in enumerate(cols_with_outliers):
                sns.boxplot(x=df_clean[col].dropna(), ax=axes[idx], color="lightcoral")
                axes[idx].set_title(f"{col} ({outlier_counts[col]:,} outliers)",
                                    fontsize=11, fontweight="bold")
            for idx in range(len(cols_with_outliers), len(axes)):
                axes[idx].axis("off")
            fig.suptitle("Outlier Inspection (Columns with Detected IQR Outliers)",
                         fontsize=15, fontweight="bold", y=1.02)
            plt.tight_layout()
            path = os.path.join(charts_dir, "outlier_boxplots.png")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved_charts.append(("Outlier Boxplots (Active Outliers Only)", "Charts/outlier_boxplots.png"))

        # Chart 6: Metric by Category
        if cat_cols and primary_metric:
            target_cat = cat_cols[0]
            fig, ax = plt.subplots(figsize=(10, 5))
            grouped = (
                df_clean.groupby(target_cat)[primary_metric]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            labels = [str(x)[:35] for x in grouped.index]
            sns.barplot(x=grouped.values, y=labels, hue=labels, palette="magma", legend=False, ax=ax)
            ax.set_title(f"Average '{primary_metric}' across '{target_cat}'", fontsize=13, fontweight="bold")
            ax.set_xlabel(f"Mean {primary_metric}")
            ax.set_ylabel(target_cat)
            plt.tight_layout()
            path = os.path.join(charts_dir, "primary_metric_by_category.png")
            fig.savefig(path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved_charts.append((f"Average {primary_metric} by {target_cat}", "Charts/primary_metric_by_category.png"))

        # Chart 7: Time-Series
        if month_cols and primary_metric:
            m_col = month_cols[0]
            monthly = df_clean[df_clean[m_col] != "NaT"].groupby(m_col)[primary_metric].sum()
            if len(monthly) >= 2:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(monthly.index, monthly.values, marker="o", color="teal", linewidth=2)
                ax.set_title(f"Time-Series Trend: Total '{primary_metric}' by {m_col}",
                             fontsize=13, fontweight="bold")
                ax.set_xlabel("Month")
                ax.set_ylabel(f"Total {primary_metric}")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                path = os.path.join(charts_dir, "monthly_trend.png")
                fig.savefig(path, bbox_inches="tight")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                saved_charts.append((f"Time-Series Trend ({primary_metric})", "Charts/monthly_trend.png"))

    print(f"✅ Step 4/4: Saved PDF Report: {pdf_path}")

    # HTML Dashboard
    html_path = os.path.join(output_dir, "EDA_Dashboard_Report.html")
    desc_html = (
        df_clean[num_cols].describe().T.round(2).to_html(classes="table")
        if num_cols else "<p>No numeric metrics found.</p>"
    )
    preview_html = df_clean.head(10).to_html(classes="table", index=False)
    insights_html = "".join(f"<li>{item}</li>" for item in insights)
    cleaning_items_html = "".join(f"<li>{item}</li>" for item in cleaning_log)
    charts_html = "".join(
        f'<div class="card"><h3>{title}</h3><img src="{rel_path}" alt="{title}"></div>'
        for title, rel_path in saved_charts
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EDA Report - {base_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f9; color: #2c3e50; margin: 0; padding: 30px; }}
        .container {{ max-width: 1200px; margin: auto; }}
        .header {{ background: #1e293b; color: white; padding: 25px; border-radius: 10px; margin-bottom: 25px; }}
        .badge {{ display: inline-block; background: #3b82f6; color: white; padding: 5px 12px; border-radius: 20px; font-size: 13px; margin-top: 8px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }}
        .metric-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; }}
        .metric-box h2 {{ margin: 0; color: #2563eb; font-size: 26px; }}
        .metric-box p {{ margin: 5px 0 0; color: #64748b; font-size: 13px; }}
        .card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 25px; overflow-x: auto; }}
        .table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        .table th, .table td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
        .table th {{ background: #f8fafc; color: #334155; }}
        img {{ max-width: 100%; height: auto; border-radius: 6px; margin-top: 10px; }}
        ul {{ line-height: 1.7; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Universal Context-Aware EDA Dashboard (v4.3)</h1>
            <p>Source: <strong>{os.path.basename(file_path)}</strong> | Cleaned CSV Copy: <strong>cleaned_{sanitize_filename(base_name)}.csv</strong></p>
            <span class="badge">Detected Context: {context['domain']}</span>
            <span class="badge">Primary Metric: {primary_metric or 'N/A'}</span>
        </div>
        <div class="grid">
            <div class="metric-box"><h2>{quality['overall_score']} / 100</h2><p>Source Quality Score</p></div>
            <div class="metric-box"><h2>{df_raw.shape[0]:,} → {df_clean.shape[0]:,}</h2><p>Rows (Raw → Cleaned)</p></div>
            <div class="metric-box"><h2>{len(saved_charts)}</h2><p>Signal-Driven Charts Built</p></div>
            <div class="metric-box"><h2>{sum(outlier_counts.values()):,}</h2><p>IQR Outliers Flagged</p></div>
        </div>
        <div class="card">
            <h3>💡 Context-Aware Statistical Insights</h3>
            <ul>{insights_html}</ul>
        </div>
        <div class="card">
            <h3>🧹 Cleaning & Organization Audit Log</h3>
            <ul>{cleaning_items_html}</ul>
        </div>
        <div class="card">
            <h3>🗂️ Organized Dataset Preview (First 10 Rows)</h3>
            {preview_html}
        </div>
        <div class="card">
            <h3>📈 Summary Statistics (Genuine Numeric Metrics Only)</h3>
            {desc_html}
        </div>
        {charts_html}
    </div>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n" + "=" * 68)
    print("🎉 UNIVERSAL PIPELINE COMPLETE (v4.3)")
    print("=" * 68)
    print(f"🧠 Detected Context : {context['domain']}")
    print(f"🎯 Primary Metric   : {primary_metric or 'N/A'}")
    print(f"📁 Output Folder    : {output_dir}")
    print(f"📄 PDF Report       : {pdf_path}")
    print(f"🌐 HTML Dashboard   : {html_path}")
    print(f"📊 Cleaned CSV      : {cleaned_csv_path}")
    print(f"📈 Relevant Charts  : {len(saved_charts)} generated")
    print(f"⭐ Quality Score    : {quality['overall_score']} / 100")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="auto_eda_pro v4.3 — Universal Context-Aware EDA & Data Organization Pipeline."
    )
    parser.add_argument("file_path", nargs="?", help="Path to input file (.csv, .json, .xlsx, .parquet).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom directory path to store output reports.")
    parser.add_argument("--missing-threshold", type=float, default=0.70,
                        help="Flag columns with missing ratio above this (default: 0.70).")
    parser.add_argument("--auto", action="store_true",
                        help="Non-interactive mode (remove dups, drop high-missing, safe-fill NaN).")
    args = parser.parse_args()

    file_input = args.file_path
    if not file_input:
        print("=" * 68)
        print("🚀 UNIVERSAL AUTOMATED EDA & ORGANIZATION PIPELINE (v4.3)")
        print("=" * 68)
        file_input = input("Enter the path to your dataset file (.csv, .json, .xlsx): ")

    if not file_input or not file_input.strip():
        print("❌ No file provided.")
    else:
        generate_eda_report(
            file_input,
            output_dir_arg=args.output_dir,
            missing_thresh=args.missing_threshold,
            auto_mode=args.auto,
        )


if __name__ == "__main__":
    main()