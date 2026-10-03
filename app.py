import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import IsolationForest


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Data Quality Monitor",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🔍 AI Data Quality Monitoring System")

st.write(
    "Upload a CSV or Excel dataset to automatically analyze "
    "data quality, missing values, duplicates, outliers and anomalies."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📁 Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)


# =========================================================
# NO FILE
# =========================================================

if uploaded_file is None:

    st.info(
        "Please upload a CSV or Excel file from the sidebar."
    )

    st.markdown("## 🚀 System Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Data Analysis")
        st.write(
            "Analyze rows, columns, data types and statistics."
        )

    with col2:
        st.subheader("🔍 Quality Detection")
        st.write(
            "Detect missing values, duplicate records and outliers."
        )

    with col3:
        st.subheader("🤖 AI Detection")
        st.write(
            "Use Machine Learning to detect unusual records."
        )

    st.stop()


# =========================================================
# READ DATASET
# =========================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_excel(uploaded_file)

except Exception as error:

    st.error(
        f"Unable to read the uploaded file: {error}"
    )

    st.stop()


# =========================================================
# BASIC DATA INFORMATION
# =========================================================

rows = len(df)

columns = len(df.columns)

total_cells = rows * columns

missing_cells = int(
    df.isnull().sum().sum()
)

duplicate_rows = int(
    df.duplicated().sum()
)


# =========================================================
# MISSING PERCENTAGE
# =========================================================

if total_cells > 0:

    missing_percentage = (
        missing_cells / total_cells
    ) * 100

else:

    missing_percentage = 0


# =========================================================
# DUPLICATE PERCENTAGE
# =========================================================

if rows > 0:

    duplicate_percentage = (
        duplicate_rows / rows
    ) * 100

else:

    duplicate_percentage = 0


# =========================================================
# QUALITY SCORE
# =========================================================

quality_score = 100

# Missing value penalty
quality_score -= min(
    missing_percentage * 0.5,
    30
)

# Duplicate penalty
quality_score -= min(
    duplicate_percentage * 0.5,
    20
)

# Empty column penalty
empty_columns = int(
    df.isnull().all().sum()
)

quality_score -= min(
    empty_columns * 5,
    20
)

quality_score = max(
    0,
    min(100, quality_score)
)


# =========================================================
# QUALITY STATUS
# =========================================================

if quality_score >= 80:

    quality_status = "🟢 Excellent"

elif quality_score >= 60:

    quality_status = "🟡 Good"

elif quality_score >= 40:

    quality_status = "🟠 Needs Improvement"

else:

    quality_status = "🔴 Poor"


# =========================================================
# DASHBOARD METRICS
# =========================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Rows",
        f"{rows:,}"
    )

with col2:
    st.metric(
        "Columns",
        f"{columns:,}"
    )

with col3:
    st.metric(
        "Missing Cells",
        f"{missing_cells:,}"
    )

with col4:
    st.metric(
        "Duplicates",
        f"{duplicate_rows:,}"
    )

with col5:
    st.metric(
        "Quality Score",
        f"{quality_score:.1f}/100"
    )


st.success(
    f"Overall Data Quality: {quality_status}"
)


# =========================================================
# DATA PREVIEW
# =========================================================

st.header("👀 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Information",
        "❌ Missing Values",
        "♻️ Duplicates",
        "📈 Outliers",
        "🤖 AI Anomalies"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("Column Information")

    info = pd.DataFrame({

        "Column":
            df.columns,

        "Data Type":
            df.dtypes.astype(str).values,

        "Missing Values":
            df.isnull().sum().values,

        "Unique Values":
            df.nunique().values

    })

    st.dataframe(
        info,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if not numeric_df.empty:

        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns found."
        )


# =========================================================
# TAB 2 - MISSING VALUES
# =========================================================

with tab2:

    st.subheader(
        "❌ Missing Value Analysis"
    )

    missing_df = pd.DataFrame({

        "Column":
            df.columns,

        "Missing Values":
            df.isnull().sum().values

    })

    if rows > 0:

        missing_df["Missing Percentage"] = (
            missing_df["Missing Values"]
            / rows
        ) * 100

    else:

        missing_df["Missing Percentage"] = 0

    missing_df = missing_df.sort_values(
        "Missing Values",
        ascending=False
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    problem_columns = missing_df[
        missing_df["Missing Values"] > 0
    ]

    if problem_columns.empty:

        st.success(
            "✅ No missing values found!"
        )

    else:

        fig = px.bar(
            problem_columns,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# TAB 3 - DUPLICATES
# =========================================================

with tab3:

    st.subheader(
        "♻️ Duplicate Record Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Duplicate Rows",
            duplicate_rows
        )

    with col2:

        st.metric(
            "Duplicate Percentage",
            f"{duplicate_percentage:.2f}%"
        )

    if duplicate_rows == 0:

        st.success(
            "✅ No duplicate records found!"
        )

    else:

        st.warning(
            f"⚠️ {duplicate_rows} duplicate records detected."
        )

        duplicate_data = df[
            df.duplicated(keep=False)
        ]

        st.dataframe(
            duplicate_data.head(50),
            use_container_width=True
        )


# =========================================================
# TAB 4 - OUTLIERS
# =========================================================

with tab4:

    st.subheader(
        "📈 Statistical Outlier Detection"
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        st.info(
            "No numerical columns available."
        )

    else:

        selected_column = st.selectbox(
            "Select numerical column",
            numeric_columns
        )

        values = df[
            selected_column
        ].dropna()

        if len(values) > 0:

            q1 = values.quantile(0.25)

            q3 = values.quantile(0.75)

            iqr = q3 - q1

            lower_limit = q1 - (
                1.5 * iqr
            )

            upper_limit = q3 + (
                1.5 * iqr
            )

            outliers = values[
                (values < lower_limit)
                |
                (values > upper_limit)
            ]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Q1",
                    f"{q1:.2f}"
                )

            with col2:

                st.metric(
                    "Q3",
                    f"{q3:.2f}"
                )

            with col3:

                st.metric(
                    "Outliers",
                    len(outliers)
                )

            fig = px.box(
                df,
                y=selected_column,
                title=f"Box Plot - {selected_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# TAB 5 - AI ANOMALIES
# =========================================================

with tab5:

    st.subheader(
        "🤖 AI-Based Anomaly Detection"
    )

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.shape[1] == 0:

        st.info(
            "At least one numerical column is required."
        )

    elif len(numeric_df) < 10:

        st.warning(
            "At least 10 numerical records are recommended."
        )

    else:

        # Replace infinite values
        numeric_df = numeric_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Fill missing values
        numeric_df = numeric_df.fillna(
            numeric_df.median()
        )

        numeric_df = numeric_df.fillna(0)

        # AI model
        model = IsolationForest(
            contamination="auto",
            random_state=42
        )

        predictions = model.fit_predict(
            numeric_df
        )

        labels = np.where(
            predictions == -1,
            "Anomaly",
            "Normal"
        )

        result_df = df.copy()

        result_df["AI_Anomaly"] = labels

        anomaly_count = int(
            (labels == "Anomaly").sum()
        )

        normal_count = int(
            (labels == "Normal").sum()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Normal Records",
                normal_count
            )

        with col2:

            st.metric(
                "AI Anomalies",
                anomaly_count
            )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        chart_data = pd.DataFrame({

            "Type": [
                "Normal",
                "Anomaly"
            ],

            "Count": [
                normal_count,
                anomaly_count
            ]

        })

        fig = px.pie(
            chart_data,
            names="Type",
            values="Count",
            title="Normal vs AI Anomalies"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# DOWNLOAD
# =========================================================

st.header("📥 Export")

csv_output = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Dataset",
    data=csv_output,
    file_name="analyzed_dataset.csv",
    mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Data Quality Monitoring System | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
