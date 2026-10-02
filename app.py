"""Autonomous Self-Healing Data Analyst Agent - Streamlit Application.
Provides a comprehensive UI for automated data ingestion, cleaning,
statistical pattern discovery, interactive visualization suite,
dedicated executive visual dashboard, no-code chart studio,
conversational analyst with real-time RAG self-healing inspection,
and executive report exporting.
"""

import os
import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from backend.data_loader import DataLoader
from backend.data_profiler import DataProfiler
from backend.pipeline import AnalysisPipeline
from backend.agent import AnalystAgent
from backend.rag_engine import rag_engine
from backend.report_generator import ReportGenerator
from backend.visualizer import Visualizer, apply_pro_styling

# Page configuration
st.set_page_config(
    page_title="Self-Healing Data Analyst Agent & Dashboard Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        font-size: 2.0rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: left;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    
    .agent-trace-box {
        background: #0F172A;
        color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        font-family: 'Consolas', monospace;
        font-size: 0.88rem;
        margin-bottom: 15px;
        border-left: 4px solid #3B82F6;
    }
    .badge-heal {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-type {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 9px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = []
if "pinned_charts" not in st.session_state:
    st.session_state.pinned_charts = []


def load_dataset_into_state(file_input, file_name, sheet_name=0):
    try:
        df, meta = DataLoader.load_file(file_input, file_name, sheet_name=sheet_name)
        st.session_state.df = df
        st.session_state.raw_df = df.copy()
        st.session_state.metadata = meta
        st.session_state.pipeline_results = None
        st.session_state.chat_history = []
        st.session_state.agent_trace = []
        st.session_state.pinned_charts = []
        return True, "Dataset loaded successfully!"
    except Exception as e:
        return False, f"Failed to load dataset: {str(e)}"


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🤖 Data Analyst Agent")
    st.caption("Autonomous • Self-Healing • RAG-Guided")

    st.markdown("---")
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose CSV, Excel (.xlsx), or JSON",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload structured data to trigger the self-healing analysis agent."
    )

    # Sheet selection if Excel
    sheet_selection = 0
    if uploaded_file and uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet_selection = st.selectbox("Select Sheet", xl.sheet_names)
        except Exception:
            pass

    if uploaded_file is not None:
        if st.session_state.metadata is None or st.session_state.metadata.get("file_name") != uploaded_file.name:
            success, msg = load_dataset_into_state(uploaded_file, uploaded_file.name, sheet_selection)
            if success:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("#### Or Try Sample Datasets")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("👥 Churn (CSV)", use_container_width=True):
            load_dataset_into_state("sample_datasets/customer_churn.csv", "customer_churn.csv")
            st.rerun()
    with col_s2:
        if st.button("📈 Sales (XLSX)", use_container_width=True):
            load_dataset_into_state("sample_datasets/sales_performance.xlsx", "sales_performance.xlsx", sheet_name="Daily_Sales")
            st.rerun()
    with col_s3:
        if st.button("🌐 Web (JSON)", use_container_width=True):
            load_dataset_into_state("sample_datasets/web_analytics.json", "web_analytics.json")
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Agent & Engine Settings")
    provider_choice = st.selectbox(
        "Intelligence Provider",
        ["Offline Heuristic Engine (Free & Instant)", "Google Gemini", "OpenAI"],
        index=0,
        help="The Heuristic Engine runs 100% locally with zero dependencies or keys. Gemini/OpenAI provide generative code synthesis."
    )

    api_key_input = ""
    provider_code = "heuristic"
    if "Gemini" in provider_choice:
        provider_code = "gemini"
        api_key_input = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    elif "OpenAI" in provider_choice:
        provider_code = "openai"
        api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    max_healing_retries = st.slider("Max Self-Healing Retries", min_value=1, max_value=6, value=4,
                                   help="Maximum code correction loops with RAG official docs upon error.")

    st.markdown("---")
    # Export options if pipeline has been run
    if st.session_state.pipeline_results is not None:
        st.subheader("📥 Export Outputs")
        p_res = st.session_state.pipeline_results
        
        # HTML Report Download
        html_rep = ReportGenerator.generate_html_report(p_res, title=f"Analysis Report - {st.session_state.metadata.get('file_name', 'Dataset')}")
        st.download_button(
            "📄 Download HTML Executive Report",
            data=html_rep,
            file_name="executive_data_analysis_report.html",
            mime="text/html",
            use_container_width=True
        )

        # Markdown Report Download
        md_rep = ReportGenerator.generate_markdown_report(p_res)
        st.download_button(
            "📝 Download Markdown Report",
            data=md_rep,
            file_name="data_analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Cleaned Data Export
        cleaned_csv = p_res["cleaned_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download Cleaned CSV",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==================== MAIN BANNER ====================
st.markdown("""
<div class="main-header">
    <h1>Autonomous Self-Healing Data Analyst Agent & Dashboard Studio</h1>
    <p>Ingests CSV, Excel, or JSON • Multi-Archetype Visualizations • Interactive Dashboard Studio • Self-Heals via Official Docs RAG</p>
</div>
""", unsafe_allow_html=True)


# Guard: Check if dataset is loaded
if st.session_state.df is None:
    st.info("👋 Welcome! Upload a CSV, Excel, or JSON dataset in the sidebar, or select one of the ready-to-use sample datasets above to begin.")
    
    st.markdown("""
    ### 🌟 How This Agent Works:
    1. **Multi-Format Ingestion**: Auto-detects delimiters, sheets, nested JSON records, and headers.
    2. **Automated Pipeline**: Complete data profiling, data health scoring, automatic cleaning, outlier detection, and correlation analysis.
    3. **14+ Visualization Archetypes**: Bar (Ranked/Grouped/Stacked), Line, Area, Donut, Treemap, Sunburst, 4D Bubble, Box, Violin, Histogram, Pareto 80/20, 2D Density Heatmap, Radar, and Funnel.
    4. **Interactive Dashboard Studio**: Executive KPI dashboard with dynamic data slicers, plus a no-code visual studio allowing non-technical users to build and pin custom charts.
    5. **Autonomous Code Generation & RAG Self-Healing**: Writes custom Python scripts, intercepts runtime exceptions, queries official Python/Pandas docs, and iteratively rewrites and executes the code until clean execution.
    """)
    st.stop()


# Profile the active dataset
df_active = st.session_state.df
metadata = st.session_state.metadata
profile = DataProfiler.profile_dataset(df_active)

# Top KPI Metric Strip
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
with kpi_col1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">File Name</div>
        <div class="kpi-val" style="font-size: 1.1rem; overflow:hidden; text-overflow:ellipsis;">{metadata.get('file_name', 'Data')}</div>
    </div>""", unsafe_allow_html=True)
with kpi_col2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Total Records</div>
        <div class="kpi-val">{profile['total_rows']:,}</div>
    </div>""", unsafe_allow_html=True)
with kpi_col3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Features / Cols</div>
        <div class="kpi-val">{profile['total_cols']}</div>
    </div>""", unsafe_allow_html=True)
with kpi_col4:
    health_color = "#10B981" if profile['health_score'] >= 85 else ("#F59E0B" if profile['health_score'] >= 60 else "#EF4444")
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Data Health Score</div>
        <div class="kpi-val" style="color: {health_color};">{profile['health_score']}%</div>
    </div>""", unsafe_allow_html=True)
with kpi_col5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Missing Values</div>
        <div class="kpi-val">{profile['missing_pct']}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Profile & Health",
    "🚀 1-Click Autonomous Pipeline",
    "📈 Interactive Visual Dashboard Studio",
    "💬 Conversational Analyst & Self-Healing",
    "📚 Official Docs RAG Knowledge Base"
])


# ==================== TAB 1: DATA PROFILE & HEALTH ====================
with tab1:
    st.subheader("📋 Dataset Inspection")
    view_option = st.radio("View Mode", ["First 15 Rows", "Summary Statistics", "Column Health Details"], horizontal=True)

    if view_option == "First 15 Rows":
        st.dataframe(df_active.head(15), use_container_width=True)
    elif view_option == "Summary Statistics":
        st.dataframe(df_active.describe(include="all").transpose(), use_container_width=True)
    else:
        # Detailed column breakdown table
        col_rows = []
        for col, d in profile["col_details"].items():
            col_rows.append({
                "Column": col,
                "Detected Type": d["type"],
                "Data Type": d["dtype"],
                "Unique Values": d["unique_count"],
                "Missing Values": f"{d['null_count']} ({d['null_pct']}%)",
                "Outliers": d.get("outliers_count", "-"),
                "Mean / Top": d.get("mean", list(d.get("top_categories", {}).keys())[:1] or "-")
            })
        st.dataframe(pd.DataFrame(col_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Automated Quality & Pattern Findings")
    for pattern in profile["key_patterns"]:
        st.markdown(f"- {pattern}")

    if profile["correlations"]:
        st.markdown("#### ⚡ Top Correlated Dimensions")
        corr_display = pd.DataFrame(profile["correlations"])[["col1", "col2", "correlation", "strength", "direction"]]
        corr_display.columns = ["Variable 1", "Variable 2", "Pearson Correlation (r)", "Strength", "Direction"]
        st.dataframe(corr_display.head(5), use_container_width=True)


# ==================== TAB 2: 1-CLICK AUTONOMOUS PIPELINE ====================
with tab2:
    st.subheader("⚡ End-to-End Autonomous Pipeline")
    st.caption("Executes the complete pipeline: Data Ingestion ➔ Smart Cleaning ➔ Statistical Discovery ➔ Multi-Archetype Visuals ➔ Executive Delivery")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_pipeline = st.button("🚀 Run Complete Pipeline", type="primary", use_container_width=True)

    if run_pipeline or st.session_state.pipeline_results is not None:
        if run_pipeline:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(msg, pct):
                status_text.markdown(f"**Step {pct}%**: {msg}")
                progress_bar.progress(pct)

            pipeline_res = AnalysisPipeline.run_full_pipeline(df_active, on_progress=update_progress)
            st.session_state.pipeline_results = pipeline_res
            time.sleep(0.4)
            status_text.empty()
            progress_bar.empty()
            st.success("✅ Complete analysis pipeline executed successfully!")

        res = st.session_state.pipeline_results

        # Executive Takeaways Section
        st.markdown("### 💡 Executive Discoveries & Key Findings")
        for takeaway in res["insights"]["takeaways"]:
            st.markdown(f"- {takeaway}")

        st.markdown("### 🎯 Recommended Strategic Decisions")
        for rec in res["insights"]["recommendations"]:
            st.markdown(f"- {rec}")

        # Interactive Charts Grid with Category Filtering
        st.markdown("### 📈 Multi-Archetype Visual Analytics")
        charts = res["charts"]
        if charts:
            categories_available = ["All"] + sorted(list(set([c.get("category", "General") for c in charts])))
            selected_cat = st.segmented_control("Filter by Visualization Category", categories_available, default="All") if hasattr(st, "segmented_control") else st.selectbox("Filter by Category", categories_available)

            filtered_charts = charts if selected_cat == "All" else [c for c in charts if c.get("category") == selected_cat]

            for i in range(0, len(filtered_charts), 2):
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    c1 = filtered_charts[i]
                    st.markdown(f"**{c1['title']}** <span class='badge-type'>{c1.get('type', 'Chart')}</span>", unsafe_allow_html=True)
                    st.plotly_chart(c1["figure"], use_container_width=True)
                    st.caption(c1["description"])
                if i + 1 < len(filtered_charts):
                    with col_c2:
                        c2 = filtered_charts[i + 1]
                        st.markdown(f"**{c2['title']}** <span class='badge-type'>{c2.get('type', 'Chart')}</span>", unsafe_allow_html=True)
                        st.plotly_chart(c2["figure"], use_container_width=True)
                        st.caption(c2["description"])
        else:
            st.info("No visual charts generated.")

        # Data Cleaning Audit
        with st.expander("🧹 View Automated Data Cleansing Audit Log", expanded=False):
            st.write("Transformations autonomously applied to raw data:")
            for log_item in res["cleaning_log"]:
                st.markdown(f"• {log_item}")


# ==================== TAB 3: INTERACTIVE VISUAL DASHBOARD STUDIO ====================
with tab3:
    st.subheader("📈 Interactive Visual Dashboard Studio")
    st.caption("Explore executive KPI dashboards with live dynamic slicers, or build custom charts across 13+ visualization types without writing a single line of code.")

    dashboard_mode = st.radio(
        "Select Studio View Mode:",
        ["🏢 Executive KPI Dashboard & Dynamic Slicers", "🎨 Custom Visual Chart Studio (No-Code Builder)", "📌 My Pinned Custom Dashboard"],
        horizontal=True
    )

    numeric_cols = profile.get("numeric_cols", [])
    categorical_cols = profile.get("categorical_cols", [])
    datetime_cols = profile.get("datetime_cols", [])

    # ----- MODE 1: EXECUTIVE KPI DASHBOARD & DYNAMIC SLICERS -----
    if "Executive" in dashboard_mode:
        st.markdown("#### 🎛️ Interactive Data Slicers & Filters")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        sliced_df = df_active.copy()

        # Slicer 1: Categorical filter
        selected_cat_filter = "All"
        if categorical_cols:
            primary_cat = categorical_cols[0]
            cat_options = ["All"] + list(df_active[primary_cat].dropna().unique())
            with filter_col1:
                selected_cat_filter = st.selectbox(f"Filter by {primary_cat}:", cat_options)
                if selected_cat_filter != "All":
                    sliced_df = sliced_df[sliced_df[primary_cat] == selected_cat_filter]

        # Slicer 2: Numeric threshold slider
        if numeric_cols:
            primary_num = numeric_cols[0]
            min_val = float(df_active[primary_num].min())
            max_val = float(df_active[primary_num].max())
            with filter_col2:
                if min_val < max_val:
                    num_range = st.slider(f"Filter {primary_num} Range:", min_value=min_val, max_value=max_val, value=(min_val, max_val))
                    sliced_df = sliced_df[(sliced_df[primary_num] >= num_range[0]) & (sliced_df[primary_num] <= num_range[1])]

        # Slicer 3: Secondary categorical or records count
        with filter_col3:
            st.metric(
                label="Active Filtered Records",
                value=f"{len(sliced_df):,} rows",
                delta=f"{round(len(sliced_df)/len(df_active)*100, 1)}% of dataset"
            )

        st.markdown("---")
        st.markdown("#### 📊 Dynamic Executive Visual Dashboard")

        dash_row1_c1, dash_row1_c2 = st.columns(2)
        with dash_row1_c1:
            # Chart A: Segment Ranking Bar
            if categorical_cols and numeric_cols:
                fig_dash_bar = Visualizer.bar_chart(
                    sliced_df, x=categorical_cols[0], y=numeric_cols[0], agg_func="mean",
                    title=f"🏆 Segment Performance: {numeric_cols[0]} by {categorical_cols[0]}"
                )
                st.plotly_chart(fig_dash_bar, use_container_width=True)
                if st.button("📌 Pin Bar Chart to Dashboard", key="pin_dash_bar"):
                    st.session_state.pinned_charts.append({"title": f"Segment Performance ({categorical_cols[0]})", "fig": fig_dash_bar})
                    st.success("Chart pinned to My Custom Dashboard!")
            elif numeric_cols:
                fig_dash_hist = Visualizer.histogram_chart(sliced_df, x=numeric_cols[0])
                st.plotly_chart(fig_dash_hist, use_container_width=True)

        with dash_row1_c2:
            # Chart B: Composition Share Donut
            if categorical_cols:
                val_c = numeric_cols[0] if numeric_cols else None
                fig_dash_donut = Visualizer.donut_chart(
                    sliced_df, names=categorical_cols[0], values=val_c,
                    title=f"🍩 Proportional Share: {categorical_cols[0]}"
                )
                st.plotly_chart(fig_dash_donut, use_container_width=True)
                if st.button("📌 Pin Donut Chart to Dashboard", key="pin_dash_donut"):
                    st.session_state.pinned_charts.append({"title": f"Share by {categorical_cols[0]}", "fig": fig_dash_donut})
                    st.success("Chart pinned to My Custom Dashboard!")

        dash_row2_c1, dash_row2_c2 = st.columns(2)
        with dash_row2_c1:
            # Chart C: Multi-Group Variance Violin / Box
            if categorical_cols and numeric_cols:
                fig_dash_violin = Visualizer.box_violin_chart(
                    sliced_df, x=categorical_cols[0], y=numeric_cols[0], chart_type="violin",
                    title=f"🎻 Variance Spread: {numeric_cols[0]} across {categorical_cols[0]}"
                )
                st.plotly_chart(fig_dash_violin, use_container_width=True)
                if st.button("📌 Pin Violin Plot to Dashboard", key="pin_dash_violin"):
                    st.session_state.pinned_charts.append({"title": f"Variance Spread ({numeric_cols[0]})", "fig": fig_dash_violin})
                    st.success("Chart pinned to My Custom Dashboard!")
            elif len(numeric_cols) >= 2:
                fig_dash_scat = px.scatter(sliced_df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}", template="plotly_white")
                st.plotly_chart(fig_dash_scat, use_container_width=True)

        with dash_row2_c2:
            # Chart D: Pareto 80/20 Analysis or Bubble
            if categorical_cols and numeric_cols:
                fig_dash_pareto = Visualizer.pareto_chart(
                    sliced_df, category_col=categorical_cols[0], value_col=numeric_cols[0],
                    title=f"⚖️ Pareto 80/20 Analysis: {categorical_cols[0]}"
                )
                st.plotly_chart(fig_dash_pareto, use_container_width=True)
                if st.button("📌 Pin Pareto Chart to Dashboard", key="pin_dash_pareto"):
                    st.session_state.pinned_charts.append({"title": f"Pareto Analysis ({categorical_cols[0]})", "fig": fig_dash_pareto})
                    st.success("Chart pinned to My Custom Dashboard!")
            elif len(numeric_cols) >= 2:
                fig_dash_density = Visualizer.density_heatmap_chart(sliced_df, x=numeric_cols[0], y=numeric_cols[1])
                st.plotly_chart(fig_dash_density, use_container_width=True)

    # ----- MODE 2: CUSTOM VISUAL CHART STUDIO (NO-CODE BUILDER) -----
    elif "Custom Visual Chart Studio" in dashboard_mode:
        st.markdown("#### 🎨 Custom Chart Studio")
        st.write("Configure dimensions, aggregation, visual archetype, and color palette with 100% no-code controls.")

        builder_col1, builder_col2, builder_col3, builder_col4 = st.columns(4)

        with builder_col1:
            chart_archetype = st.selectbox(
                "Visualization Type",
                [
                    "Bar Chart (Ranked / Grouped)",
                    "Line Chart (Temporal Trend)",
                    "Cumulative Area Chart",
                    "Donut & Pie Chart (Composition)",
                    "Hierarchical Treemap",
                    "Sunburst Radial Hierarchy",
                    "4D Bubble Chart (X, Y, Size, Color)",
                    "Box Plot (Outlier Analysis)",
                    "Violin Plot (Density & Spread)",
                    "Histogram (Marginal Box)",
                    "Pareto Analysis (80/20 Rule)",
                    "2D Density Heatmap",
                    "Funnel Conversion Chart"
                ]
            )

        with builder_col2:
            all_cols = list(df_active.columns)
            x_col = st.selectbox("X-Axis / Primary Dimension", all_cols, index=0)
            y_col = st.selectbox("Y-Axis / Metric Column", all_cols, index=min(1, len(all_cols)-1))

        with builder_col3:
            color_opts = ["None"] + all_cols
            color_choice = st.selectbox("Color / Grouping Dimension", color_opts, index=0)
            color_col = None if color_choice == "None" else color_choice

            size_opts = ["None"] + numeric_cols
            size_choice = st.selectbox("Size Dimension (Bubble / Treemap)", size_opts, index=0)
            size_col = None if size_choice == "None" else size_choice

        with builder_col4:
            agg_choice = st.selectbox("Aggregation Function", ["sum", "mean", "count", "median", "none"], index=0)
            theme_choice = st.selectbox("Color Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "presentation"], index=0)

        custom_chart_title = st.text_input("Custom Chart Title", value=f"{chart_archetype}: {y_col} across {x_col}")

        # Render custom chart dynamically
        fig_custom = None
        try:
            if "Bar Chart" in chart_archetype:
                fig_custom = Visualizer.bar_chart(df_active, x=x_col, y=y_col, color=color_col, agg_func=agg_choice if agg_choice != "none" else None, title=custom_chart_title)
            elif "Line Chart" in chart_archetype:
                fig_custom = Visualizer.line_chart(df_active, x=x_col, y=y_col, color=color_col, title=custom_chart_title)
            elif "Area Chart" in chart_archetype:
                fig_custom = Visualizer.area_chart(df_active, x=x_col, y=y_col, color=color_col, title=custom_chart_title)
            elif "Donut" in chart_archetype:
                fig_custom = Visualizer.donut_chart(df_active, names=x_col, values=y_col if y_col in numeric_cols else None, title=custom_chart_title)
            elif "Treemap" in chart_archetype:
                path_cols = [x_col] if not color_col else [x_col, color_col]
                fig_custom = Visualizer.treemap_chart(df_active, path=path_cols, values=y_col if y_col in numeric_cols else numeric_cols[0], title=custom_chart_title)
            elif "Sunburst" in chart_archetype:
                path_cols = [x_col] if not color_col else [x_col, color_col]
                fig_custom = Visualizer.sunburst_chart(df_active, path=path_cols, values=y_col if y_col in numeric_cols else numeric_cols[0], title=custom_chart_title)
            elif "Bubble" in chart_archetype:
                fig_custom = Visualizer.bubble_chart(df_active, x=x_col, y=y_col, size=size_col, color=color_col, title=custom_chart_title)
            elif "Box Plot" in chart_archetype:
                fig_custom = Visualizer.box_violin_chart(df_active, x=x_col, y=y_col, chart_type="box", color=color_col, title=custom_chart_title)
            elif "Violin" in chart_archetype:
                fig_custom = Visualizer.box_violin_chart(df_active, x=x_col, y=y_col, chart_type="violin", color=color_col, title=custom_chart_title)
            elif "Histogram" in chart_archetype:
                fig_custom = Visualizer.histogram_chart(df_active, x=x_col, color=color_col, title=custom_chart_title)
            elif "Pareto" in chart_archetype:
                fig_custom = Visualizer.pareto_chart(df_active, category_col=x_col, value_col=y_col, title=custom_chart_title)
            elif "2D Density" in chart_archetype:
                fig_custom = Visualizer.density_heatmap_chart(df_active, x=x_col, y=y_col, title=custom_chart_title)
            elif "Funnel" in chart_archetype:
                fig_custom = Visualizer.funnel_chart(df_active, stage_col=x_col, value_col=y_col if y_col in numeric_cols else None, title=custom_chart_title)

            if fig_custom:
                fig_custom.update_layout(template=theme_choice)
                st.plotly_chart(fig_custom, use_container_width=True)

                act_col1, act_col2 = st.columns([1, 4])
                with act_col1:
                    if st.button("📌 Pin Chart to Dashboard", key="btn_pin_custom"):
                        st.session_state.pinned_charts.append({"title": custom_chart_title, "fig": fig_custom})
                        st.success("Chart pinned successfully to My Custom Dashboard!")

                with act_col2:
                    chart_html = pio.to_html(fig_custom, full_html=True, include_plotlyjs="cdn")
                    st.download_button(
                        "📥 Download Chart as Standalone HTML",
                        data=chart_html,
                        file_name=f"{custom_chart_title.replace(' ', '_').lower()}.html",
                        mime="text/html"
                    )
        except Exception as e:
            st.error(f"Could not generate {chart_archetype} with selected columns: {str(e)}")
            st.info("Tip: Ensure numeric columns are selected for Y-Axis/Size, and categorical columns for X-Axis/Donut/Treemap.")

    # ----- MODE 3: MY PINNED CUSTOM DASHBOARD -----
    elif "Pinned" in dashboard_mode:
        st.markdown("#### 📌 My Pinned Custom Dashboard")
        st.caption("Custom visualizations pinned from the Studio or Executive Slicers into a single consolidated view.")

        if not st.session_state.pinned_charts:
            st.info("No charts pinned yet! Build a chart in the Custom Visual Chart Studio or Executive Dashboard and click '📌 Pin Chart to Dashboard'.")
        else:
            p_cols = st.columns(2)
            for idx, p in enumerate(st.session_state.pinned_charts):
                with p_cols[idx % 2]:
                    st.markdown(f"**{p['title']}**")
                    st.plotly_chart(p["fig"], use_container_width=True)

            if st.button("🗑️ Clear All Pinned Charts"):
                st.session_state.pinned_charts = []
                st.rerun()


# ==================== TAB 4: CONVERSATIONAL ANALYST & SELF-HEALING ====================
with tab4:
    st.subheader("💬 Natural Language Conversational Analyst")
    st.caption("Ask any question in plain English. The agent writes code, executes it, queries official docs upon errors, and self-heals in real time.")

    # Recommended Prompt Chips
    st.markdown("**Suggested Questions for this Dataset:**")
    cand_chips = [
        "What is the distribution and outlier spread of numeric variables?",
        "Create a treemap decomposing volume across categorical dimensions",
        "Generate a donut chart showing composition share percentages",
        "Show a violin plot comparing variance across segments",
        "Perform a pareto 80/20 analysis on key drivers"
    ]
    if profile.get("categorical_cols") and profile.get("numeric_cols"):
        cand_chips[0] = f"Break down average {profile['numeric_cols'][0]} by {profile['categorical_cols'][0]}"

    chip_cols = st.columns(len(cand_chips))
    selected_chip = None
    for idx, chip in enumerate(cand_chips):
        with chip_cols[idx]:
            if st.button(chip, key=f"chip_{idx}", use_container_width=True):
                selected_chip = chip

    user_query = st.text_input(
        "Enter your analytical question:",
        value=selected_chip if selected_chip else "",
        placeholder="e.g. Show me a treemap of sales by region, or a violin plot of charges by contract..."
    )

    col_ask, col_clear = st.columns([1, 5])
    with col_ask:
        trigger_analysis = st.button("🔍 Analyze & Self-Heal", type="primary", use_container_width=True)

    if trigger_analysis and user_query:
        agent = AnalystAgent(
            api_key=api_key_input if api_key_input else None,
            provider=provider_code
        )

        step_logs = []
        with st.spinner("Agent is formulating code and analyzing..."):
            trace_expander = st.expander("🛠️ Live Self-Healing Agent Trace & RAG Inspector", expanded=True)

            def agent_callback(step_info):
                step_logs.append(step_info)
                stage = step_info.get("stage")
                msg = step_info.get("message")
                details = step_info.get("details", {})

                with trace_expander:
                    if stage == "planning":
                        st.markdown(f"🧠 **[Plan]** {msg}")
                    elif stage == "executing":
                        st.markdown(f"⚡ **[Execution]** {msg}")
                    elif stage == "error_detected":
                        st.markdown(f"<span class='badge-heal'>⚠️ Exception Captured: {details.get('error_type')}</span>", unsafe_allow_html=True)
                        st.markdown(f"`{details.get('error_message')}` (Line {details.get('failing_line')}: `{details.get('failing_code')}`)")
                    elif stage == "rag_retrieval":
                        st.markdown("📚 **[RAG Knowledge Base Query]** Retrieved Official Docs:")
                        for d in details.get("docs", []):
                            st.markdown(f"  - **{d['title']}** *({d['category']})*")
                    elif stage == "healing":
                        st.markdown(f"🩹 **[Self-Healing]** {msg}")
                    elif stage == "success":
                        st.markdown(f"<span class='badge-success'>✅ {msg}</span>", unsafe_allow_html=True)

            analysis_result = agent.run_analysis(
                df=df_active,
                query=user_query,
                max_retries=max_healing_retries,
                on_step_callback=agent_callback
            )

        # Store in chat history
        st.session_state.chat_history.append({
            "query": user_query,
            "result": analysis_result
        })

    # Display Chat Results
    if st.session_state.chat_history:
        st.markdown("---")
        for item in reversed(st.session_state.chat_history):
            q = item["query"]
            res = item["result"]

            st.markdown(f"#### ❓ Question: *{q}*")
            
            if res["success"]:
                # Insights
                st.markdown("##### 💡 Analytical Insights:")
                for ins in res["insights"]:
                    st.markdown(f"- {ins}")

                # Charts
                if res["figures"]:
                    for fig in res["figures"]:
                        st.plotly_chart(fig, use_container_width=True)

                # Healed Code Display
                with st.expander(f"💻 View Synthesized Python Code (Attempts: {res['total_attempts']})", expanded=False):
                    if len(res.get("healing_history", [])) > 0:
                        st.info(f"Self-healed across {len(res['healing_history'])} error recovery loop(s) using RAG docs.")
                    st.code(res["final_code"], language="python")
            else:
                st.error(f"Analysis failed: {res.get('error_type')}: {res.get('error_message')}")
                if res.get("final_code"):
                    st.code(res["final_code"], language="python")


# ==================== TAB 5: RAG KNOWLEDGE BASE EXPLORER ====================
with tab5:
    st.subheader("📚 RAG Official Documentation Knowledge Base")
    st.caption("Official Python, Pandas, and Plotly documentation indexed for automated error diagnostics and self-healing.")

    rag_search = st.text_input("Test Knowledge Retrieval (e.g. 'treemap values', 'pareto secondary_y', 'KeyError', 'SettingWithCopy'):")
    if rag_search:
        results = rag_engine.retrieve(rag_search, top_k=3)
        st.write(f"Found {len(results)} relevant official doc topics:")
        for r in results:
            with st.expander(f"📘 {r['title']} [{r['category']}]", expanded=True):
                st.markdown(r["doc_content"])
                if r.get("example_bad"):
                    st.markdown(f"**❌ Antipattern:** `{r['example_bad']}`")
                if r.get("example_good"):
                    st.markdown(f"**✅ Verified Fix:** `{r['example_good']}`")
    else:
        st.write("Browse pre-indexed official documentation topics:")
        for doc in rag_engine.doc_chunks:
            with st.expander(f"📖 {doc['title']} ({doc['category']})"):
                st.markdown(doc["doc_content"])
                if doc.get("example_bad"):
                    st.code(doc["example_bad"], language="python")
                if doc.get("example_good"):
                    st.code(doc["example_good"], language="python")
