import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import os

# --- 1. 全局配置 ---
st.set_page_config(
    page_title="Chicago Crime Intel",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. 状态管理 ---
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Welcome'
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = 2024

# --- 3. CSS 样式 ---
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .launch-card {
        background-color: white; padding: 40px; border-radius: 20px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.1); text-align: center;
        max-width: 600px; margin: 0 auto;
    }
    div.metric-container {
        background-color: white; padding: 15px 20px; border-radius: 10px;
        border-left: 5px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1 { color: #1f2937; font-family: 'Inter', sans-serif; }
    .js-plotly-plot .plotly .modebar { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. 数据加载函数 (只定义，不运行) ---
@st.cache_data
def load_data(year):
    possible_files = [f"chicago_crime_{year}.csv.zip", f"chicago_crime_{year}.zip"]
    search_dirs = [".", "split_data_by_year"]
    
    found_path = None
    for d in search_dirs:
        for f in possible_files:
            test_path = os.path.join(d, f)
            if os.path.exists(test_path):
                found_path = test_path
                break
        if found_path: break

    if found_path:
        import zipfile
        try:
            with zipfile.ZipFile(found_path, 'r') as z:
                # 过滤 Mac 隐藏文件
                csv_files = [name for name in z.namelist() if name.endswith('.csv') and not name.startswith('__MACOSX')]
                if csv_files:
                    with z.open(csv_files[0]) as f:
                        cols = ['Date', 'Primary Type', 'Description', 'Arrest', 'District', 'Latitude', 'Longitude', 'Location Description']
                        df = pd.read_csv(f, usecols=cols)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df['Month_Num'] = df['Date'].dt.month
                        df['Hour'] = df['Date'].dt.hour
                        df['DayOfWeek'] = df['Date'].dt.day_name()
                        return df
        except Exception as e:
            st.error(f"Error reading zip file: {e}")
            return None
    return None

# ❌ 删除了这里原本错误的 if df is None 代码块，因为它会导致 NameError

# ==========================================
# 📺 场景 A: 启动页 (Landing Page)
# ==========================================

if st.session_state.app_mode == 'Welcome':
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="launch-card">
            <h1 style="font-size: 3em; margin-bottom: 10px;">🚔 Chicago Crime Intel</h1>
            <p style="color: #6b7280; font-size: 1.2em; margin-bottom: 30px;">
                IT5006 Phase 1: Interactive Crime Analytics System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- 🔍 修复版扫描逻辑：必须显式查找 .csv.zip ---
        available_years = []
        # 根据你的截图，数据在 apps/dashboard/split_data_by_year 下
        # 程序运行在 apps/dashboard 下，所以相对路径是 split_data_by_year
        search_dirs = [".", "split_data_by_year"] 
        
        for y in range(2014, 2025):
            found = False
            for d in search_dirs:
                # 这里就是你之前缺失的逻辑：检查 .csv.zip
                if os.path.exists(os.path.join(d, f"chicago_crime_{y}.csv.zip")) or \
                   os.path.exists(os.path.join(d, f"chicago_crime_{y}.zip")) or \
                   os.path.exists(os.path.join(d, f"chicago_crime_{y}.csv")):
                    found = True
                    break
            if found:
                available_years.append(y)
        
        # 如果没扫到，保底 2024
        if not available_years: available_years = [2024]
        
        st.markdown("### 📅 Select Analysis Year")
        
        # --- 🎨 强制显示漂亮的红色滑块 ---
        # 只要有数据，就用 Select Slider
        if len(available_years) > 1:
            chosen_year = st.select_slider(
                "Select Year", 
                options=sorted(available_years), 
                value=available_years[-1], 
                label_visibility="collapsed"
            )
        else:
            # 只有一年也显示滑块样式，保持美观统一
            chosen_year = st.select_slider(
                "Select Year",
                options=available_years,
                value=available_years[0],
                label_visibility="collapsed"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button(f"🚀 Launch Dashboard ({chosen_year})", type="primary", use_container_width=True):
            st.session_state.selected_year = chosen_year
            st.session_state.app_mode = 'Dashboard'
            st.rerun()

    st.markdown("<br><br><p style='text-align: center; color: #9ca3af;'>© Team 22 | Powered by Streamlit</p>", unsafe_allow_html=True)
    
        # 修复 RangeError
        if len(available_years) > 1:
            chosen_year = st.select_slider("Select Year", options=sorted(available_years), value=available_years[-1], label_visibility="collapsed")
        else:
            chosen_year = st.selectbox("Select Year", options=available_years, label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"🚀 Launch Dashboard ({chosen_year})", type="primary", use_container_width=True):
            st.session_state.selected_year = chosen_year
            st.session_state.app_mode = 'Dashboard'
            st.rerun()

    st.markdown("<br><br><p style='text-align: center; color: #9ca3af;'>© Team 22 | Powered by Streamlit</p>", unsafe_allow_html=True)

# ==========================================
# 📊 场景 B: 主仪表盘 (Dashboard)
# ==========================================
elif st.session_state.app_mode == 'Dashboard':
    year = st.session_state.selected_year
    
    # ✅ 正确逻辑 1: 先加载数据，把结果赋值给 df
    df = load_data(year)
    
    # ✅ 正确逻辑 2: 拿到 df 后，立刻检查它是不是空的
    if df is None or df.empty:
        st.error(f"❌ 无法加载 {year} 年的数据。")
        st.info("💡 请检查 GitHub 仓库中是否上传了对应的 .zip 文件，且文件名正确 (chicago_crime_YYYY.csv.zip)。")
        if st.button("← 返回首页"):
            st.session_state.app_mode = 'Welcome'
            st.rerun()
        st.stop() # 停止运行，防止后面报错

    # ✅ 正确逻辑 3: 数据没问题了，才开始渲染页面
    # --- 侧边栏 ---
    with st.sidebar:
        if st.button("← Back to Home"):
            st.session_state.app_mode = 'Welcome'
            st.rerun()
        st.divider()
        st.title(f"🎛️ Controls ({year})")
        st.success(f"Loaded: {len(df):,} rows")
        
        all_types = sorted(df['Primary Type'].unique())
        default_types = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'ASSAULT']
        sel_types = st.multiselect("Filter Type", all_types, default=[x for x in default_types if x in all_types])
        
        districts = sorted([int(x) for x in df['District'].dropna().unique()])
        sel_districts = st.multiselect("Police District (Optional)", districts, default=[])
        arrest = st.radio("Arrest Status", ["All", "Yes", "No"], horizontal=True)

    # --- 过滤数据 ---
    mask = df['Primary Type'].isin(sel_types)
    if sel_districts: mask = mask & (df['District'].isin(sel_districts))
    if arrest == "Yes": mask = mask & (df['Arrest'] == True)
    if arrest == "No": mask = mask & (df['Arrest'] == False)
    filtered_df = df[mask]

    # --- 页面主内容 ---
    st.title(f"Chicago Crime Intelligence: {year}")
    
    def metric_card(title, value, sub, color):
        st.markdown(f"""
        <div class="metric-container" style="border-left: 5px solid {color};">
            <p style="font-size:14px; color:#6b7280; margin:0;">{title}</p>
            <p style="font-size:26px; font-weight:700; color:#111827; margin:5px 0;">{value}</p>
            <p style="font-size:12px; color:{color}; margin:0;">{sub}</p>
        </div>
        """, unsafe_allow_html=True)

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1: metric_card("Total Incidents", f"{len(filtered_df):,}", "Volume", "#3b82f6")
    with r1c2: metric_card("Arrest Rate", f"{(filtered_df['Arrest'].mean()*100 if not filtered_df.empty else 0):.1f}%", "Efficiency", "#10b981")
    with r1c3: 
        loc = filtered_df['Location Description'].mode()[0] if not filtered_df.empty else "N/A"
        metric_card("Top Location", loc[:15]+"...", "Risk Zone", "#f59e0b")
    with r1c4: 
        hour = filtered_df['Hour'].mode()[0] if not filtered_df.empty else "N/A"
        metric_card("Peak Hour", f"{hour}:00", "High Alert", "#ef4444")

    st.markdown("---")

    c_map, c_charts = st.columns([1.8, 1])
    with c_map:
        st.subheader("📍 Spatial Distribution")
        if not filtered_df.empty:
            map_data = filtered_df.dropna(subset=['Latitude','Longitude'])
            if len(map_data) > 20000: 
                map_data = map_data.sample(20000)
                st.caption(f"⚠️ Displaying random 20,000 points (out of {len(filtered_df)}) for performance.")
            
            st.pydeck_chart(pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(latitude=41.85, longitude=-87.65, zoom=10),
                layers=[pdk.Layer("ScatterplotLayer", data=map_data, get_position='[Longitude, Latitude]', get_color='[200, 30, 0, 160]', get_radius=40, pickable=True)],
                tooltip={"text": "{Primary Type}\n{Description}"}
            ))
        else: st.warning("No data available for map.")

    with c_charts:
        st.subheader("📈 Monthly Trend")
        if not filtered_df.empty:
            trend = filtered_df.groupby('Month_Num').size().reset_index(name='Count')
            import calendar
            trend['Month'] = trend['Month_Num'].apply(lambda x: calendar.month_abbr[x])
            st.plotly_chart(px.area(trend, x='Month', y='Count', markers=True).update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0)), use_container_width=True)
        
        st.subheader("📊 Crime Types")
        if not filtered_df.empty:
            top = filtered_df['Primary Type'].value_counts().head(5).reset_index()
            top.columns=['Type','Count']
            st.plotly_chart(px.bar(top, x='Count', y='Type', orientation='h', color='Count').update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), showlegend=False), use_container_width=True)
            
    # --- Heatmap (回归) ---
    st.markdown("---")
    st.subheader("🗓️ Temporal Heatmap")
    if not filtered_df.empty:
        heat = filtered_df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='Counts')
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_heat = px.density_heatmap(heat, x='Hour', y='DayOfWeek', z='Counts', category_orders={'DayOfWeek': days}, color_continuous_scale='Reds', nbinsx=24, nbinsy=7)
        fig_heat.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), xaxis=dict(dtick=1))
        st.plotly_chart(fig_heat, use_container_width=True)
