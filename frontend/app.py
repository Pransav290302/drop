

import streamlit as st
import pandas as pd
from pathlib import Path
import io
from typing import Dict, Any, Optional
import logging
import plotly.graph_objects as go
import plotly.express as px


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from frontend.config import config
    from frontend.utils.api_client import api_client
except ImportError:

    import os
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from frontend.config import config
    from frontend.utils.api_client import api_client


st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%); font-family: 'Inter',sans-serif;}
#MainMenu, footer, header {visibility:hidden;}
.ai-header { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:2.5rem 2rem; border-radius:1.5rem;margin-bottom:2rem;box-shadow:0 16px 40px #667eea45;text-shadow:0 2px 22px #000a;}
.ai-header h1 {font-size:2.5rem; font-weight:700; color:white; margin:0;}
.ai-header p {font-size:1.2rem; color:rgba(255,255,255,0.96); margin:0.5rem 0 0;}
.sidebar-glass {background:rgba(255,255,255,0.04); border-radius:1.2rem; padding:1rem 1.4rem; box-shadow:0 4px 32px #667eea30;}
.metric-card, .glass-section, .success-box, .error-box {background:rgba(255,255,255,0.065);backdrop-filter:blur(6px);border-radius:0.9rem;padding:1.2rem;margin:0.6rem 0;box-shadow:0 2px 16px #2222;}
.success-box {border:1.5px solid #43e97bb7;}
.error-box {border:1.5px solid #e74c3cbb;}
.stButton>button {border-radius:0.6rem;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;font-weight:600;}
.stButton>button:hover {transform:translateY(-2px);box-shadow:0 8px 28px #667eeaba;}
.stat-card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:1rem;padding:1.2rem 1.5rem;color:white;margin:0.5em 0;box-shadow:0 6px 18px #667eea22;}
.stat-card h3 {font-weight:700; font-size:1.7rem;}
.stMetric label {color:#a0aec0 !important;}
.stMetric [data-testid="stMetricValue"] {font-weight:700; color:white !important;}
.stDataFrame {background: rgba(255,255,255,0.04)!important; border-radius:1rem;}
select, .stSelectbox>div>div>div {background:rgba(255,255,255,0.08)!important;color:white!important;}

.hero-header {
    background: linear-gradient(135deg, #1c1f45 0%, #202a5c 46%, #1c1f45 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 1.5rem;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 20px 45px rgba(18, 21, 56, 0.55);
    display: flex;
    align-items: center;
    gap: 1rem;
}
.hero-logo {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.hero-logo .logo-icon {
    width: 60px;
    height: 60px;
    border-radius: 16px;
    background: linear-gradient(145deg, #667eea, #764ba2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    box-shadow: 0 12px 24px rgba(102,126,234,0.45);
}
.hero-logo .logo-text {
    font-family: 'Inter', sans-serif;
}
.hero-logo .logo-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
}
.hero-logo .logo-subtitle {
    color: rgba(255,255,255,0.75);
    font-size: 0.95rem;
}
.custom-navbar {
    background: rgba(30, 35, 70, 0.8);
    border-radius: 1rem;
    padding: 0.75rem 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    gap: 0.5rem;
    align-items: center;
}
.custom-navbar .stButton {
    flex: 1;
}
.custom-navbar .stButton > button {
    background: rgba(40, 45, 80, 0.6) !important;
    border: none !important;
    border-radius: 0.6rem !important;
    padding: 0.65rem 1rem !important;
    color: #d3d8ff !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: none !important;
    width: 100% !important;
    height: auto !important;
    min-height: auto !important;
}
.custom-navbar .stButton > button:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.custom-navbar .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}
.custom-navbar .stButton > button:active {
    transform: translateY(0) !important;
}
.custom-navbar button[key="clear_session_top"] {
    background: rgba(60, 65, 100, 0.6) !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 0.8rem !important;
}
.custom-navbar button[key="clear_session_top"]:hover {
    background: rgba(100, 105, 140, 0.8) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ai-header">
    <h1>🤖 DropSmart</h1>
    <p>Product & Price Intelligence for Dropshipping Sellers</p>
</div>
""", unsafe_allow_html=True)


if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None
if "results" not in st.session_state:
    st.session_state.results = None
if "selected_sku" not in st.session_state:
    st.session_state.selected_sku = None


if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home / Upload"

nav_options = [
    "🏠 Home / Upload",
    "📊 Dashboard",
    "🔍 Product Detail",
    "📈 Product Insights",
    "📥 Export CSV",
]

st.markdown(f'''
<script>
function setActiveNav() {{
    const activePage = "{st.session_state.current_page}";
    const buttons = document.querySelectorAll('.custom-navbar button:not([key="clear_session_top"])');
    buttons.forEach(btn => {{
        if (btn.textContent.trim() === activePage) {{
            btn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            btn.style.color = '#ffffff';
            btn.style.fontWeight = '600';
            btn.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.5)';
        }}
    }});
}}
if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', setActiveNav);
}} else {{
    setActiveNav();
}}
setTimeout(setActiveNav, 100);
</script>
<div class="custom-navbar">
''', unsafe_allow_html=True)

nav_cols = st.columns([1, 1, 1, 1, 1, 2])

for idx, nav_item in enumerate(nav_options):
    with nav_cols[idx]:
        if st.button(nav_item, key=f"nav_{idx}", use_container_width=True):
            st.session_state.current_page = nav_item
            st.rerun()

with nav_cols[5]:
    try:
        api_status = "✅ Connected" if api_client.health_check() else "❌ Disconnected"
        st.markdown(f'<div style="color: #a0aec0; font-size: 0.85rem; padding-top: 0.5rem; text-align: right;">{api_status}</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div style="color: #a0aec0; font-size: 0.85rem; padding-top: 0.5rem; text-align: right;">❌ Error</div>', unsafe_allow_html=True)
    if st.button("🔄 Clear", key="clear_session_top", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.current_page




if page == "🏠 Home / Upload":
    st.header("📤 Upload Product Data")
    st.markdown("Upload your supplier Excel file to get started with product analysis.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help="Upload a file with product data including SKU, product_name, cost, price, shipping_cost, lead_time_days, and availability"
    )
    
    if uploaded_file is not None:
        # Store uploaded file
        st.session_state.uploaded_file = uploaded_file
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("Type", uploaded_file.type or "application/vnd.ms-excel")
        
        # Upload button
        if st.button("📤 Upload to Server", type="primary", width="stretch"):
            with st.spinner("Uploading file..."):
                try:
                    # Validate filename
                    if not uploaded_file.name:
                        st.error("❌ File must have a name. Please select a file with a filename.")
                        st.stop()
                    
                    # Validate file extension
                    file_ext = Path(uploaded_file.name).suffix.lower()
                    if file_ext not in [".xlsx", ".xls"]:
                        st.error(f"❌ Invalid file type '{file_ext}'. Please upload an Excel file (.xlsx or .xls)")
                        st.stop()
                    
                    # Read file bytes - ensure we're at the beginning
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    
                    # Validate file is not empty
                    if len(file_bytes) == 0:
                        st.error("❌ File is empty. Please upload a non-empty file.")
                        st.stop()
                    
                    # Upload to API
                    upload_response = api_client.upload_file(file_bytes, uploaded_file.name)
                    
                    # Store file_id
                    st.session_state.file_id = upload_response["file_id"]
                    
                    st.success(f"✅ File uploaded successfully!")
                    st.info(f"**File ID:** {st.session_state.file_id}")
                    st.info(f"**Total Rows:** {upload_response['total_rows']}")
                    
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    logger.error(f"Upload error: {e}", exc_info=True)
        

        if st.session_state.file_id:
            st.markdown("---")
            st.subheader("🔍 Schema Validation")
            
            if st.button("✅ Validate Schema", type="primary", width="stretch"):
                with st.spinner("Validating schema..."):
                    try:
                        validation_result = api_client.validate_schema(st.session_state.file_id)
                        st.session_state.validation_result = validation_result
                        
                        if validation_result["is_valid"]:
                            st.success("✅ Schema is valid!")
                            if validation_result.get("warnings"):
                                st.warning(f"⚠️ {len(validation_result['warnings'])} warnings found")
                        else:
                            st.error(f"❌ Schema validation failed with {len(validation_result['errors'])} errors")
                        
                      
                        with st.expander("📋 Validation Details", expanded=not validation_result["is_valid"]):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Summary**")
                                st.write(f"- Total Rows: {validation_result['total_rows']}")
                                st.write(f"- Total Columns: {validation_result['total_columns']}")
                                st.write(f"- Valid: {'✅ Yes' if validation_result['is_valid'] else '❌ No'}")
                            
                            with col2:
                                st.write("**Missing Fields**")
                                if validation_result.get("missing_required_fields"):
                                    st.error("Required:")
                                    for field in validation_result["missing_required_fields"]:
                                        st.write(f"  - {field}")
                                if validation_result.get("missing_optional_fields"):
                                    st.warning("Optional:")
                                    for field in validation_result["missing_optional_fields"]:
                                        st.write(f"  - {field}")
                            
                            
                            if validation_result.get("errors"):
                                st.write("**Errors**")
                                for error in validation_result["errors"]:
                                    st.error(f"- {error.get('field', 'Unknown')}: {error.get('message', 'Unknown error')}")
                            
                           
                            if validation_result.get("warnings"):
                                st.write("**Warnings**")
                                for warning in validation_result["warnings"]:
                                    st.warning(f"- {warning}")
                        
                        
                        if validation_result["is_valid"]:
                            st.markdown("---")
                            if st.button("🚀 Process Products", type="primary", width="stretch"):
                                with st.spinner("Processing products with ML models... This may take a moment."):
                                    try:
                                        results = api_client.get_results(st.session_state.file_id)
                                        st.session_state.results = results
                                        st.success(f"✅ Processed {results['total_products']} products successfully!")
                                        st.balloons()
                                        st.info("👉 Navigate to **Dashboard** to view results")
                                    except Exception as e:
                                        st.error(f"❌ Processing failed: {str(e)}")
                                        logger.error(f"Processing error: {e}", exc_info=True)
                    
                    except Exception as e:
                        st.error(f"❌ Validation failed: {str(e)}")
                        logger.error(f"Validation error: {e}", exc_info=True)

elif page == "📊 Dashboard":
    st.header("📊 Product Dashboard")
    st.markdown("View ranked products with viability scores, recommended prices, and risk assessments.")
    
    if st.session_state.file_id is None:
        st.warning("⚠️ Please upload a file first from the Home page.")
        st.info("👉 Go to **Home / Upload** to upload your Excel file")
    elif st.session_state.results is None:
        st.warning("⚠️ No results available. Please process the uploaded file.")
        if st.button("🚀 Process Products", type="primary"):
            with st.spinner("Processing products..."):
                try:
                    results = api_client.get_results(st.session_state.file_id)
                    st.session_state.results = results
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)
    else:
        results = st.session_state.results
        
       
        st.subheader("📈 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", results.get("total_products", 0))
        with col2:
            high_viability = sum(1 for r in results.get("results", []) if r.get("viability_class", "").lower() == "high")
            st.metric("High Viability", high_viability)
        with col3:
            high_risk = sum(1 for r in results.get("results", []) if r.get("stockout_risk_level", "").lower() == "high")
            st.metric("High Risk", high_risk)
        with col4:
            results_list = results.get("results", [])
            if results_list:
                avg_viability = sum(r.get("viability_score", 0) for r in results_list) / len(results_list)
                st.metric("Avg Viability", f"{avg_viability:.2%}")
            else:
                st.metric("Avg Viability", "N/A")
        
        st.markdown("---")
        
      
        st.subheader("📋 Ranked Products")
        
       
        df_data = []
        for result in results.get("results", []):
            df_data.append({
                "Rank": result.get("rank", 0),
                "SKU": result.get("sku", "N/A"),
                "Product Name": result.get("product_name", "N/A"),
                "Viability Score": result.get("viability_score", 0.0),
                "Viability Class": result.get("viability_class", "low").title(),
                "Recommended Price": f"${result.get('recommended_price', 0.0):.2f}",
                "Current Price": f"${result.get('current_price', 0.0):.2f}",
                "Margin %": f"{result.get('margin_percent', 0.0):.1f}%",
                "Stockout Risk": result.get("stockout_risk_level", "low").title(),
                "Risk Score": f"{result.get('stockout_risk_score', 0.0):.2f}",
                "Cluster ID": result.get("cluster_id", "N/A") if result.get("cluster_id") is not None else "N/A",
            })
        
        if not df_data:
            st.warning("No product data available")
        else:
            df = pd.DataFrame(df_data)
            
            
            col1, col2, col3 = st.columns(3)
            with col1:
                viability_options = ["High", "Medium", "Low"]
                viability_filter = st.multiselect(
                    "Filter by Viability",
                    options=viability_options,
                    default=[]
                )
            with col2:
                risk_options = ["High", "Medium", "Low"]
                risk_filter = st.multiselect(
                    "Filter by Risk",
                    options=risk_options,
                    default=[]
                )
            with col3:
                search_sku = st.text_input("Search SKU", "")
            
           
            filtered_df = df.copy()
            if viability_filter:
                filtered_df = filtered_df[filtered_df["Viability Class"].isin(viability_filter)]
            if risk_filter:
                filtered_df = filtered_df[filtered_df["Stockout Risk"].isin(risk_filter)]
            if search_sku:
                filtered_df = filtered_df[filtered_df["SKU"].str.contains(search_sku, case=False, na=False)]
            
           
            st.dataframe(
                filtered_df,
                width="stretch",
                height=400,
                hide_index=True
            )
            
            st.caption(f"Showing {len(filtered_df)} of {len(df)} products")
            
           
            st.markdown("---")
            st.subheader("🔍 View Product Details")
            
            sku_options = [r.get("sku", "N/A") for r in results.get("results", []) if r.get("sku")]
            if sku_options:
                selected_sku = st.selectbox(
                    "Select a product to view details",
                    options=sku_options,
                    key="detail_sku_selector"
                )
                
                if selected_sku:
                    st.session_state.selected_sku = selected_sku
                    if st.button("View Details", type="primary"):
                        st.info("👉 Navigate to **Product Detail** page to see full analysis")
            else:
                st.warning("No products available for selection")

elif page == "🔍 Product Detail":
    st.header("🔍 Product Detail Analysis")
    
    if st.session_state.results is None:
        st.warning("⚠️ No results available. Please process a file first.")
    elif st.session_state.selected_sku is None:
        st.warning("⚠️ Please select a product from the Dashboard.")
        st.info("👉 Go to **Dashboard** and select a product to view details")
    else:
        results = st.session_state.results
        selected_sku = st.session_state.selected_sku
        
      
        product = None
        for r in results.get("results", []):
            if r.get("sku") == selected_sku:
                product = r
                break
        
        if product is None:
            st.error("Product not found")
        else:
          
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(product.get("product_name", "Unknown Product"))
                st.write(f"**SKU:** {product.get('sku', 'N/A')}")
                st.write(f"**Rank:** #{product.get('rank', 'N/A')}")
            
            with col2:
              
                viability_class = product.get("viability_class", "low").lower()
                viability_color = {
                    "high": "🟢",
                    "medium": "🟡",
                    "low": "🔴"
                }
                st.metric(
                    "Viability",
                    f"{viability_color.get(viability_class, '⚪')} {viability_class.title()}",
                    f"{product.get('viability_score', 0.0):.2%}"
                )
            
            st.markdown("---")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Viability Score", f"{product.get('viability_score', 0.0):.2%}")
            with col2:
                st.metric("Recommended Price", f"${product.get('recommended_price', 0.0):.2f}")
            with col3:
                st.metric("Margin %", f"{product.get('margin_percent', 0.0):.1f}%")
            with col4:
                risk_level = product.get("stockout_risk_level", "low").lower()
                risk_color = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }
                st.metric(
                    "Stockout Risk",
                    f"{risk_color.get(risk_level, '⚪')} {risk_level.title()}"
                )
            
            st.markdown("---")
            
          
            st.subheader("💰 Pricing Analysis")
            col1, col2, col3 = st.columns(3)
            
            current_price = product.get("current_price", 0.0)
            recommended_price = product.get("recommended_price", 0.0)
            
            with col1:
                st.write("**Current Price**")
                st.write(f"${current_price:.2f}")
            with col2:
                st.write("**Recommended Price**")
                st.write(f"${recommended_price:.2f}")
            with col3:
                price_change = recommended_price - current_price
                price_change_pct = (price_change / current_price * 100) if current_price > 0 else 0
                st.write("**Change**")
                if price_change >= 0:
                    st.write(f"🔼 +${price_change:.2f} ({price_change_pct:+.1f}%)")
                else:
                    st.write(f"🔽 ${price_change:.2f} ({price_change_pct:.1f}%)")
            
            st.markdown("---")
            
    
            st.subheader("⚠️ Risk Analysis")
            st.write(f"**Risk Score:** {product.get('stockout_risk_score', 0.0):.2%}")
            st.write(f"**Risk Level:** {product.get('stockout_risk_level', 'low').title()}")
            
            cluster_id = product.get("cluster_id")
            if cluster_id is not None:
                st.write(f"**Cluster ID:** {cluster_id}")
            
            st.markdown("---")
            
            st.subheader("📊 Feature Importance (SHAP)")
            st.info("💡 SHAP values show how each feature contributes to the viability prediction")
            
         
            shap_values = product.get("shap_values")
            base_value = product.get("base_value")
            
            if shap_values and isinstance(shap_values, dict):
              
                shap_items = list(shap_values.items())
                shap_items.sort(key=lambda x: abs(x[1]), reverse=True)
                
              
                top_features = shap_items[:15]
                
                if top_features:
                    # Prepare data for visualization
                    feature_names = [item[0] for item in top_features]
                    feature_values = [item[1] for item in top_features]
                    
                   
                    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in feature_values]
                    
                    # Create horizontal bar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=feature_values,
                        y=feature_names,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{v:+.4f}" for v in feature_values],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Top 15 Feature Contributions (SHAP Values)",
                        xaxis_title="SHAP Value",
                        yaxis_title="Feature",
                        height=500,
                        showlegend=False,
                        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                        margin=dict(l=150, r=50, t=50, b=50)
                    )
                    
                    st.plotly_chart(fig, width="stretch")
                    
                  
                    if base_value is not None:
                        st.caption(f"Base value (expected output): {base_value:.4f}")
                    
                    with st.expander("📋 View All Feature Contributions"):
                        shap_df = pd.DataFrame({
                            "Feature": feature_names,
                            "SHAP Value": feature_values,
                            "Impact": ["Positive" if v >= 0 else "Negative" for v in feature_values]
                        })
                        st.dataframe(shap_df, width="stretch")
                else:
                    st.warning("No SHAP values available for this product.")
            else:
                
                try:
                   
                    viability_response = api_client.predict_viability([product])
                    if viability_response and "predictions" in viability_response:
                        pred = viability_response["predictions"][0]
                        if pred.get("shap_values"):
                            st.info("🔄 Fetching SHAP values from API...")
                           
                            product["shap_values"] = pred["shap_values"]
                            product["base_value"] = pred.get("base_value")
                            st.rerun()
                        else:
                            st.info("SHAP values are not available for this product. The model may not support SHAP explanations.")
                    else:
                        st.info("SHAP values are not available for this product. The model may not support SHAP explanations.")
                except Exception as e:
                    logger.warning(f"Failed to fetch SHAP values: {e}")
                    st.info("SHAP values are not available for this product. The model may not support SHAP explanations.")

elif page == "📈 Product Insights":
    st.header("📈 Product Insights Dashboard")
    st.markdown("Visualize cost structure and performance metrics for any processed product.")
    
    if st.session_state.results is None:
        st.warning("⚠️ No results available. Please process a file first from the Home page.")
    else:
        product_entries = st.session_state.results.get("results", [])
        if not product_entries:
            st.info("No products available yet. Upload and process a file to unlock insights.")
        else:
            product_options = {
                f"{item.get('sku', 'SKU')} — {item.get('product_name', 'Product')}": item
                for item in product_entries
            }
            
            selected_label = st.selectbox(
                "Select a product to visualize",
                list(product_options.keys()),
                key="insights_product_selector"
            )
            selected_product = product_options[selected_label]
            
            viability_score = float(selected_product.get("viability_score", 0) or 0)
            stockout_score = float(selected_product.get("stockout_risk_score", 0) or 0)
            margin_percent = float(selected_product.get("margin_percent", 0) or 0)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(
                    "Viability Score",
                    f"{viability_score:.2f}",
                    selected_product.get("viability_class", "low").title()
                )
            with col_b:
                st.metric(
                    "Stockout Risk",
                    f"{stockout_score:.2f}",
                    selected_product.get("stockout_risk_level", "low").title()
                )
            with col_c:
                st.metric("Margin %", f"{margin_percent:.2f}%")
            
            cost = float(selected_product.get("cost", 0) or 0)
            shipping = float(selected_product.get("shipping_cost", 0) or 0)
            duties = float(selected_product.get("duties", 0) or 0)
            recommended_price = float(
                selected_product.get("recommended_price",
                                     selected_product.get("current_price", 0)) or 0
            )
            landed_cost = cost + shipping + duties
            profit_component = max(recommended_price - landed_cost, 0)
            
            breakdown_entries = [
                {"Component": "Product Cost", "Value": max(cost, 0)},
                {"Component": "Shipping", "Value": max(shipping, 0)},
                {"Component": "Duties", "Value": max(duties, 0)},
                {"Component": "Profit (Recommended Price)", "Value": profit_component},
            ]
            breakdown_df = pd.DataFrame(breakdown_entries)
            breakdown_df = breakdown_df[breakdown_df["Value"] > 0]
            
         
            if len(breakdown_df) < 2:
                estimated_cost = max(recommended_price - profit_component, 0)
                if estimated_cost <= 0 and recommended_price > 0:
                    estimated_cost = max(
                        recommended_price * (1 - (margin_percent / 100 if margin_percent else 0)),
                        0
                    )
                
                if estimated_cost > 0:
                    breakdown_df = pd.DataFrame([
                        {"Component": "Landed Cost (Est.)", "Value": estimated_cost},
                        {"Component": "Profit (Recommended Price)", "Value": max(profit_component, 0.01)},
                    ])
                elif recommended_price > 0:
                    breakdown_df = pd.DataFrame([
                        {"Component": "Revenue", "Value": recommended_price * 0.5},
                        {"Component": "Profit (Recommended Price)", "Value": recommended_price * 0.5},
                    ])
                else:
                    breakdown_df = pd.DataFrame([{"Component": "N/A", "Value": 1}])
            
            normalized_metrics = pd.DataFrame([
                {"Metric": "Viability Score", "Value": max(min(viability_score, 1), 0)},
                {"Metric": "Stockout Risk", "Value": max(min(stockout_score, 1), 0)},
                {"Metric": "Margin (0-1)", "Value": max(min(margin_percent / 100, 1), 0)},
            ])
            
            col_pie, col_bar = st.columns(2)
            with col_pie:
                st.subheader("Cost & Profit Breakdown")
                pie_colors = (
                    px.colors.sequential.Purples
                    + px.colors.sequential.Blues
                    + px.colors.sequential.Greens
                )
                fig_pie = px.pie(
                    breakdown_df,
                    names="Component",
                    values="Value",
                    hole=0.45,
                    color_discrete_sequence=pie_colors[: len(breakdown_df)]
                )
                fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, width="stretch")
            
            with col_bar:
                st.subheader("Performance Snapshot")
                fig_bar = px.bar(
                    normalized_metrics,
                    x="Metric",
                    y="Value",
                    range_y=[0, 1],
                    color="Metric",
                    color_discrete_sequence=px.colors.sequential.Magma
                )
                fig_bar.update_layout(
                    showlegend=False,
                    yaxis=dict(tickformat=".0%", title="Normalized Value"),
                    margin=dict(t=10, b=40, l=10, r=10)
                )
                st.plotly_chart(fig_bar, width="stretch")
            
            with st.expander("📋 Product Snapshot"):
                product_df = pd.DataFrame.from_dict(selected_product, orient="index", columns=["Value"])
                product_df["Value"] = product_df["Value"].astype("string")
                st.dataframe(product_df, width="stretch")

elif page == "📥 Export CSV":
    st.header("📥 Export Results to CSV")
    
    if st.session_state.file_id is None:
        st.warning("⚠️ No file uploaded. Please upload a file first.")
        st.info("👉 Go to **Home / Upload** to upload your Excel file")
    else:
        st.markdown("Export your analysis results to CSV for import into Amazon, Shopify, or your ERP system.")
        
        if st.session_state.results:
            results = st.session_state.results
            
          
            csv_data = []
            for result in results.get("results", []):
                csv_data.append({
                    "SKU": result.get("sku", ""),
                    "Product Name": result.get("product_name", ""),
                    "Rank": result.get("rank", 0),
                    "Viability Score": result.get("viability_score", 0.0),
                    "Viability Class": result.get("viability_class", "low"),
                    "Recommended Price": result.get("recommended_price", 0.0),
                    "Current Price": result.get("current_price", 0.0),
                    "Margin %": result.get("margin_percent", 0.0),
                    "Stockout Risk Score": result.get("stockout_risk_score", 0.0),
                    "Stockout Risk Level": result.get("stockout_risk_level", "low"),
                    "Cluster ID": result.get("cluster_id", "") if result.get("cluster_id") is not None else "",
                })
            
            if csv_data:
                df_export = pd.DataFrame(csv_data)
                
              
                st.subheader("📋 Export Preview")
                st.dataframe(df_export.head(10), width="stretch")
                st.caption(f"Total rows: {len(df_export)}")
            else:
                st.warning("No data available for preview")
        else:
            st.info("💡 Results will be fetched from the server when you export.")
        
        st.markdown("---")
      
        if st.button("📥 Export CSV from Server", type="primary", width="stretch"):
            with st.spinner("Generating CSV file..."):
                try:
                  
                    csv_bytes = api_client.export_csv(st.session_state.file_id)
                    
                  
                    file_id_short = st.session_state.file_id[:8] if st.session_state.file_id else "unknown"
                    filename = f"dropsmart_results_{file_id_short}.csv"
                    

                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv_bytes,
                        file_name=filename,
                        mime="text/csv",
                        type="primary",
                        width="stretch",
                        key="csv_download_button"
                    )
                    
                    st.success("✅ CSV file generated successfully!")
                    st.info("💡 Click the download button above to save the CSV file. It can be imported into Amazon, Shopify, or your ERP system.")
                    
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
                    logger.error(f"CSV export error: {e}", exc_info=True)
                    st.info("💡 Make sure you have processed the file first. Go to **Home / Upload** and click 'Process Products'.")
        

        if st.session_state.results:
            st.markdown("---")
            st.subheader("Alternative: Download from Cached Results")
            st.caption("This uses locally cached results. For the latest data, use the server export above.")
            

            csv_data = []
            for result in st.session_state.results.get("results", []):
                csv_data.append({
                    "SKU": result.get("sku", ""),
                    "Product Name": result.get("product_name", ""),
                    "Rank": result.get("rank", 0),
                    "Viability Score": result.get("viability_score", 0.0),
                    "Viability Class": result.get("viability_class", "low"),
                    "Recommended Price": result.get("recommended_price", 0.0),
                    "Current Price": result.get("current_price", 0.0),
                    "Margin %": result.get("margin_percent", 0.0),
                    "Stockout Risk Score": result.get("stockout_risk_score", 0.0),
                    "Stockout Risk Level": result.get("stockout_risk_level", "low"),
                    "Cluster ID": result.get("cluster_id", "") if result.get("cluster_id") is not None else "",
                })
            
            if csv_data:
                df_export = pd.DataFrame(csv_data)
                csv_string = df_export.to_csv(index=False)
                csv_bytes_local = csv_string.encode('utf-8')
                
                file_id_short = st.session_state.file_id[:8] if st.session_state.file_id else "unknown"
                filename_local = f"dropsmart_results_{file_id_short}.csv"
                
                st.download_button(
                    label="📥 Download from Cache",
                    data=csv_bytes_local,
                    file_name=filename_local,
                    mime="text/csv",
                    width="stretch",
                    key="csv_download_cache"
                )


st.markdown("---")
st.caption(f"Product Intelligence v{config.PAGE_TITLE} | API: {config.API_BASE_URL}")
