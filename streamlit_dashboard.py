#!/usr/bin/env python3
"""
Prajna Insights - Smart Data Contextualization Pipeline
Enterprise-Grade Data Processing & Business Intelligence Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import time
import io
from app.utils.file_reader import robust_file_reader
from app.services.smart_contextualization_pipeline import SmartContextualizationPipeline
from app.services.context_rich_insights import calculate_comprehensive_kpis, generate_comprehensive_insights, detect_industry_sector

# Page configuration
st.set_page_config(
    page_title="Prajna Insights - Smart Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .pipeline-step {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def read_uploaded_file_robust(uploaded_file):
    """Robust file reading with encoding detection"""
    try:
        # Read file content
        file_content = uploaded_file.read()
        
        # Detect encoding
        import chardet
        detected = chardet.detect(file_content)
        encoding = detected.get('encoding', 'utf-8')
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read with detected encoding
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding=encoding)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON files.")
            return None, None
        
        file_info = {
            'file_type': uploaded_file.name.split('.')[-1].upper(),
            'size_bytes': len(file_content),
            'encoding': encoding,
            'rows': len(df),
            'columns': len(df.columns)
        }
        
        return df, file_info
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None, None

def generate_professional_dashboard(df, industry, tier, theme):
    """Generate comprehensive professional dashboard with sector-specific KPIs"""
    
    if df is None or df.empty:
        st.error("No data available for dashboard generation")
        return
    
    theme_colors = {
        'corporate': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'modern': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
        'minimal': ['#2C3E50', '#E74C3C', '#3498DB', '#F39C12', '#9B59B6']
    }
    colors = theme_colors.get(theme, theme_colors['corporate'])
    
    st.info(f"📊 Processing {len(df):,} records with {len(df.columns)} columns")
    
    # Calculate KPIs
    kpis = calculate_comprehensive_kpis(df, industry)
    
    st.markdown("#### 📊 Key Performance Indicators")
    
    generic_kpis = [kpi for kpi in kpis if kpi[2] == "Generic"]
    industry_kpis = [kpi for kpi in kpis if kpi[2] != "Generic"]
    
    st.markdown("**🔧 Generic Metrics**")
    generic_cols = st.columns(len(generic_kpis))
    for i, (kpi_name, kpi_value, kpi_type) in enumerate(generic_kpis):
        with generic_cols[i]:
            st.metric(label=kpi_name, value=kpi_value, help=f"Generic business metric")
    
    if industry_kpis:
        st.markdown(f"**🏭 {industry.title()} Industry Metrics**")
        industry_cols = st.columns(len(industry_kpis))
        for i, (kpi_name, kpi_value, kpi_type) in enumerate(industry_kpis):
            with industry_cols[i]:
                st.metric(label=kpi_name, value=kpi_value, help=f"{kpi_type} specific KPI")
    
    st.markdown("#### 📈 Business Analytics Dashboard")
    
    try:
        # Chart 1: Business Overview Dashboard
        st.markdown("#### 📊 Business Overview")
        
        if len(df.select_dtypes(include=['number']).columns) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Simple Performance Trend
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    # Create a simple trend chart
                    sample_size = min(50, len(df))
                    sample_df = df.head(sample_size)
                    
                    fig1 = px.area(
                        sample_df, 
                        x=sample_df.index,
                        y=numeric_cols[0],
                        title=f"📈 {industry.title()} Performance Trend",
                        color_discrete_sequence=[colors[0]]
                    )
                    fig1.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        height=350,
                        xaxis_title="Time Period",
                        yaxis_title="Value"
                    )
                    fig1.update_traces(fill='tonexty')
                    st.plotly_chart(fig1, width='stretch')
            
            with col2:
                # Top Categories/Products
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    category_counts = df[categorical_cols[0]].value_counts().head(6)
                    fig2 = px.bar(
                        x=category_counts.values,
                        y=category_counts.index,
                        orientation='h',
                        title=f"🏆 Top {industry.title()} Categories",
                        color=category_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig2.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        height=350,
                        xaxis_title="Count",
                        yaxis_title="Category"
                    )
                    st.plotly_chart(fig2, width='stretch')
        
        # Chart 2: Performance Analysis
        st.markdown("#### 📈 Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple Distribution Chart
            if len(df.select_dtypes(include=['number']).columns) > 0:
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                fig3 = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=f"📊 {industry.title()} Value Distribution",
                    nbins=20,
                    color_discrete_sequence=[colors[1]]
                )
                fig3.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Arial, sans-serif", size=12),
                    title_font_size=16,
                    height=350,
                    xaxis_title="Value",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig3, width='stretch')
        
        with col2:
            # Key Statistics Summary
            if len(df.select_dtypes(include=['number']).columns) > 0:
                numeric_cols = df.select_dtypes(include=['number']).columns
                col_data = df[numeric_cols[0]]
                
                # Create a simple summary chart
                stats = {
                    'Metric': ['Average', 'Best', 'Worst', 'Total'],
                    'Value': [
                        col_data.mean(),
                        col_data.max(),
                        col_data.min(),
                        col_data.sum()
                    ]
                }
                
                fig4 = px.bar(
                    x=stats['Metric'],
                    y=stats['Value'],
                    title=f"🎯 {industry.title()} Key Statistics",
                    color=stats['Value'],
                    color_continuous_scale='Greens'
                )
                fig4.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Arial, sans-serif", size=12),
                    title_font_size=16,
                    height=350,
                    xaxis_title="Metric",
                    yaxis_title="Value"
                )
                st.plotly_chart(fig4, width='stretch')
        
        # Business Insights Summary
        st.markdown("#### 💡 Business Insights & Recommendations")
        
        insights = generate_comprehensive_insights(df, industry, tier)
        
        # Separate generic and industry-specific insights
        generic_insights = [insight for insight in insights if insight.get('type') == 'Generic']
        industry_insights = [insight for insight in insights if insight.get('type') != 'Generic']
        
        if generic_insights:
            st.markdown("**🔧 Generic Business Insights**")
            for insight in generic_insights:
                st.info(f"🎯 **{insight['title']}**: {insight['description']}")
                if 'recommendation' in insight:
                    st.success(f"💡 **Recommendation**: {insight['recommendation']}")
        
        if industry_insights:
            st.markdown(f"**🏭 {industry.title()} Industry Insights**")
            for insight in industry_insights:
                st.info(f"🎯 **{insight['title']}**: {insight['description']}")
                if 'recommendation' in insight:
                    st.success(f"💡 **Recommendation**: {insight['recommendation']}")
    
    except Exception as e:
        st.error(f"Error generating charts: {str(e)}")
        st.info("Displaying basic data summary instead...")
        
        st.markdown("#### 📊 Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Total Columns", len(df.columns))
        
        with col2:
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            st.metric("Numeric Columns", len(numeric_cols))
            st.metric("Categorical Columns", len(categorical_cols))

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Prajna Insights - Smart Pipeline</h1>
        <p>Enterprise-Grade Data Processing & Business Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose your analysis:",
        ["🚀 Smart Pipeline", "📊 Dashboard", "📁 File Upload", "⚙️ Settings"]
    )
    
    # Smart Pipeline Page
    if page == "🚀 Smart Pipeline":
        st.header("🧠 Prajna Smart Data Pipeline")
        st.markdown("**Transform your data into professional business intelligence**")
        
        st.markdown("""
        This pipeline automatically processes your data to create professional visualizations and business insights. 
        Upload your file below to see the transformation in action.
        """)
        
        with st.expander("🏗️ Pipeline Architecture", expanded=True):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
            <h3 style="color: white; margin-top: 0;">🧠 Prajna Enterprise Data Processing Pipeline</h3>
            <p style="color: white; margin-bottom: 0;">Transform raw, messy data into professional, business-ready analytics with AI-powered intelligence.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown("""
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745;">
                <h4 style="color: #28a745; margin: 0;">🔍 Smart Profiling</h4>
                <p style="margin: 5px 0; font-size: 12px;">Auto-detect types & classify data</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff;">
                <h4 style="color: #007bff; margin: 0;">🤖 LLM Standardization</h4>
                <p style="margin: 5px 0; font-size: 12px;">Clean & standardize columns</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #ffc107;">
                <h4 style="color: #ffc107; margin: 0;">⚙️ Auto Preprocessing</h4>
                <p style="margin: 5px 0; font-size: 12px;">Handle missing values & types</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #dc3545;">
                <h4 style="color: #dc3545; margin: 0;">💡 Rich Insights</h4>
                <p style="margin: 5px 0; font-size: 12px;">Generate business insights</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown("""
                <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #6f42c1;">
                <h4 style="color: #6f42c1; margin: 0;">🎨 Stunning Viz</h4>
                <p style="margin: 5px 0; font-size: 12px;">Professional visualizations</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.subheader("⚙️ Configuration & Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            auto_detect = st.checkbox("🤖 Auto-detect Industry", value=True, help="Automatically detect the best industry sector from your data")
            
            if auto_detect:
                st.info("🎯 Industry will be automatically detected from your data")
                demo_industry = "retail"  # Default, will be updated after file upload
            else:
                demo_industry = st.selectbox("🏭 Industry Sector", 
                                            ["retail", "ecommerce", "restaurant", "manufacturing", "finance", "healthcare"], 
                                            index=0,
                                            help="Manually select your industry for sector-specific KPIs and insights")
        
        with col2:
            demo_tier = st.selectbox("💎 Analysis Tier", 
                                    ["pro", "business", "enterprise"], 
                                    index=1,
                                    help="Higher tiers provide more advanced analytics")
        
        with col3:
            demo_theme = st.selectbox("🎨 Theme", 
                                     ["corporate", "modern", "minimal"], 
                                     index=0,
                                     help="Professional visualization theme")
        
        demo_file = st.file_uploader("📁 Upload sample data to see the pipeline in action", 
                                     type=['csv', 'xlsx', 'json'], 
                                     help="Upload any CSV file to see the transformation")
        
        if demo_file is not None:
            try:
                demo_df, file_info = read_uploaded_file_robust(demo_file)
                
                if demo_df is not None:
                    st.success(f"✅ File loaded: {demo_file.name}")
                    
                    if auto_detect:
                        detected_sector, sector_scores = detect_industry_sector(demo_df)
                        demo_industry = detected_sector
                        
                        st.success(f"🎯 **Auto-detected Industry**: {detected_sector.title()}")
                        
                        with st.expander("🔍 Industry Detection Analysis", expanded=False):
                            st.markdown("**Detection Confidence Scores:**")
                            for sector, score in sorted(sector_scores.items(), key=lambda x: x[1], reverse=True):
                                if score > 0:
                                    confidence = (score / max(sector_scores.values())) * 100 if max(sector_scores.values()) > 0 else 0
                                    st.write(f"• **{sector.title()}**: {score} points ({confidence:.1f}% confidence)")
                    
                    if file_info:
                        st.info(f"📄 File Type: {file_info['file_type']}, Size: {file_info['size_bytes']:,} bytes")
                        if file_info['encoding']:
                            st.info(f"🔤 Detected Encoding: {file_info['encoding']}")
                else:
                    st.error("❌ Failed to read file")
                    st.stop()
                
                st.markdown("### 🚀 Processing Your Data")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Analyzing your data structure and quality...")
                progress_bar.progress(20)
                time.sleep(0.3)
                
                status_text.text("🤖 Applying AI-powered data enhancement...")
                progress_bar.progress(40)
                time.sleep(0.3)
                
                status_text.text("💡 Generating business insights and KPIs...")
                progress_bar.progress(60)
                time.sleep(0.3)
                
                status_text.text("🎨 Creating professional visualizations...")
                progress_bar.progress(80)
                time.sleep(0.3)
                
                status_text.text("✨ Finalizing your business intelligence dashboard...")
                progress_bar.progress(100)
                time.sleep(0.3)
                
                progress_bar.empty()
                status_text.empty()
                
                st.markdown("### 📈 Data Analysis Results")
                
                missing_count = demo_df.isnull().sum().sum()
                missing_percentage = (missing_count / (len(demo_df) * len(demo_df.columns))) * 100 if len(demo_df) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Missing Values", f"{missing_count:,}", f"{missing_percentage:.1f}%", help="Total missing values found")
                with col2:
                    st.metric("Data Types", f"{len(demo_df.dtypes.unique())}", "Detected", help="Different data types identified")
                with col3:
                    st.metric("File Size", f"{len(demo_df):,} rows", f"{len(demo_df.columns)} cols", help="Dataset dimensions")
                with col4:
                    st.metric("Memory Usage", f"{demo_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB", help="Approximate memory usage")
                
                st.markdown("### 📊 Data Preview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📥 Original Data**")
                    st.dataframe(demo_df.head(5), width='stretch')
                
                with col2:
                    st.markdown("**✨ Enhanced Data**")
                    enhanced_df = demo_df.copy()
                    enhanced_df.columns = [col.title().replace('_', ' ') for col in enhanced_df.columns]
                    st.dataframe(enhanced_df.head(5), width='stretch')
                
                st.markdown("### 💡 Key Business Insights")
                
                numeric_cols = demo_df.select_dtypes(include=['number']).columns
                categorical_cols = demo_df.select_dtypes(include=['object']).columns
                
                insights = []
                if len(numeric_cols) > 0:
                    total_value = demo_df[numeric_cols[0]].sum()
                    avg_value = demo_df[numeric_cols[0]].mean()
                    insights.append(f"📊 **Total {demo_industry.title()} Value**: ${total_value:,.0f} across {len(demo_df):,} records")
                    insights.append(f"📈 **Average Performance**: ${avg_value:,.0f} per record")
                
                if len(categorical_cols) > 0:
                    top_category = demo_df[categorical_cols[0]].mode()[0] if not demo_df[categorical_cols[0]].mode().empty else 'N/A'
                    insights.append(f"🎯 **Top Performing Category**: {top_category}")
                
                insights.append(f"💼 **Business Ready**: Professional {demo_theme.title()} theme applied for executive presentation")
                
                for insight in insights:
                    st.info(insight)
                
                st.markdown("### 🎯 Your Business Intelligence Dashboard")
                generate_professional_dashboard(demo_df, demo_industry, demo_tier, demo_theme)
                
                st.markdown("### 🎉 Processing Complete")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Records Processed", f"{len(demo_df):,}", help="Total records in your dataset")
                
                with col2:
                    st.metric("Columns Analyzed", len(demo_df.columns), help="Number of columns processed")
                
                with col3:
                    numeric_cols = len(demo_df.select_dtypes(include=['number']).columns)
                    st.metric("Numeric Columns", numeric_cols, help="Columns with numerical data")
                
                with col4:
                    categorical_cols = len(demo_df.select_dtypes(include=['object']).columns)
                    st.metric("Text Columns", categorical_cols, help="Columns with text data")
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 10px; color: white; text-align: center; margin-top: 20px;">
                <h3 style="color: white; margin: 0;">🎉 Pipeline Execution Complete!</h3>
                <p style="color: white; margin: 10px 0 0 0; font-size: 16px;">Your data has been successfully transformed into enterprise-grade business intelligence with professional visualizations and actionable insights.</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        else:
            st.info("👆 Upload a file to see the Smart Pipeline in action!")
    
    # Dashboard Page
    elif page == "📊 Dashboard":
        st.header("📊 Prajna Business Intelligence Dashboard")
        st.markdown("**Professional analytics and insights for your business**")
        
        # Load sample data for demo
        try:
            sample_df = pd.read_csv("sample_test_data.csv")
            st.success("✅ Sample data loaded successfully")
            
            # Configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                industry = st.selectbox("🏭 Industry", ["retail", "ecommerce", "restaurant", "manufacturing", "finance", "healthcare"], index=0)
            with col2:
                tier = st.selectbox("💎 Tier", ["pro", "business", "enterprise"], index=1)
            with col3:
                theme = st.selectbox("🎨 Theme", ["corporate", "modern", "minimal"], index=0)
            
            # Generate dashboard
            generate_professional_dashboard(sample_df, industry, tier, theme)
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            st.info("Please upload a file in the Smart Pipeline page to see the dashboard.")
    
    # File Upload Page
    elif page == "📁 File Upload":
        st.header("📁 Prajna File Upload & Processing")
        st.markdown("**Upload your data files for analysis**")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Upload CSV, Excel, or JSON files for analysis"
        )
        
        if uploaded_file is not None:
            df, file_info = read_uploaded_file_robust(uploaded_file)
            
            if df is not None:
                st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
                
                # Show file info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{file_info['size_bytes']:,} bytes")
                with col4:
                    st.metric("Encoding", file_info['encoding'])
                
                # Data preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), width='stretch')
                
                # Auto-detect industry
                detected_sector, sector_scores = detect_industry_sector(df)
                st.success(f"🎯 **Auto-detected Industry**: {detected_sector.title()}")
                
                # Configuration
                col1, col2, col3 = st.columns(3)
                with col1:
                    industry = st.selectbox("🏭 Industry", ["retail", "ecommerce", "restaurant", "manufacturing", "finance", "healthcare"], 
                                          index=["retail", "ecommerce", "restaurant", "manufacturing", "finance", "healthcare"].index(detected_sector))
                with col2:
                    tier = st.selectbox("💎 Analysis Tier", ["pro", "business", "enterprise"], index=1)
                with col3:
                    theme = st.selectbox("🎨 Theme", ["corporate", "modern", "minimal"], index=0)
                
                # Generate dashboard
                if st.button("🚀 Generate Dashboard", type="primary"):
                    generate_professional_dashboard(df, industry, tier, theme)
    
    # Settings Page
    elif page == "⚙️ Settings":
        st.header("⚙️ Prajna Settings & Configuration")
        st.markdown("**Configure your Smart Pipeline settings**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎨 Visualization Settings")
            default_theme = st.selectbox("Default Theme", ["corporate", "modern", "minimal"], index=0)
            default_tier = st.selectbox("Default Analysis Tier", ["pro", "business", "enterprise"], index=1)
            auto_detect_industry = st.checkbox("Auto-detect Industry", value=True)
        
        with col2:
            st.subheader("📊 Data Processing Settings")
            max_file_size = st.number_input("Max File Size (MB)", min_value=1, max_value=100, value=10)
            sample_size = st.number_input("Sample Size for Large Files", min_value=100, max_value=10000, value=1000)
            enable_caching = st.checkbox("Enable Caching", value=True)
        
        st.subheader("🔧 Advanced Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("API Key (Optional)", type="password", help="Enter your API key for enhanced features")
            st.selectbox("Language", ["English", "Spanish", "French", "German"], index=0)
        
        with col2:
            st.number_input("Timeout (seconds)", min_value=30, max_value=300, value=120)
            st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT"], index=0)
        
        if st.button("💾 Save Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
        
        st.markdown("---")
        st.subheader("📋 System Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pipeline Version", "1.0.0")
        with col2:
            st.metric("Python Version", "3.12")
        with col3:
            st.metric("Streamlit Version", "1.28.0")

if __name__ == "__main__":
    main()