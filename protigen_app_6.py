import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PROFESSIONAL UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="ProtiGen | Advanced Clinical AI", page_icon="🧬", layout="wide")

# Custom CSS for that "High-End Medical" look
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stButton>button { border-radius: 10px; height: 3.5em; background-color: #238636; color: white; border: none; font-weight: bold; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #2ea043; border: 1px solid white; }
    .metric-card { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; text-align: center; }
    .report-header { background: linear-gradient(90deg, #1f6feb 0%, #238636 100%); padding: 2px; border-radius: 10px; margin-bottom: 20px; }
    .report-content { background-color: #0d1117; padding: 20px; border-radius: 8px; }
    div[data-testid="stExpander"] { border: 1px solid #30363d; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. AI CORE ENGINE
# ==========================================
class DeepANN(nn.Module):
    def __init__(self, num_features):
        super(DeepANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_clinical_assets():
    # Load Scaler
    scaler = joblib.load("protigen_scaler.pkl")
    # Define Model
    model = DeepANN(scaler.n_features_in_)
    model.load_state_dict(torch.load("best_Deep_ANN.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    # Get Bacteria Names from Database safely
    conn = sqlite3.connect('ProtiGen_DB.sqlite')
    temp_df = pd.read_sql_query("SELECT * FROM Microbiome_Profiles LIMIT 1", conn)
    bacteria_names = [c for c in temp_df.columns if c not in ['Patient_ID', 'Outcome_Label']]
    conn.close()
    
    return model, scaler, bacteria_names

try:
    model, scaler, bacteria_list = load_clinical_assets()
except Exception as e:
    st.error(f"Critical Error Loading Assets: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("ProtiGen v4.0")
    st.markdown("*Precision Oncology Suite*")
    st.markdown("---")
    
    if 'current_page' not in st.session_state: st.session_state.current_page = "Home"
    
    if st.sidebar.button("🏠 Home Dashboard"): st.session_state.current_page = "Home"
    if st.sidebar.button("🔬 Molecular Analytics"): st.session_state.current_page = "Analytics"
    if st.sidebar.button("➕ Register New Patient"): st.session_state.current_page = "New"
    
    st.markdown("---")
    st.caption("AI Model: **Deep Neural Network**")
    st.caption("Accuracy: **97.10%**")
    st.caption("Status: ✅ **Connected**")

# ==========================================
# 4. PAGE: HOME DASHBOARD
# ==========================================
if st.session_state.current_page == "Home":
    st.title("🏥 Clinical Intelligence Dashboard")
    st.write("Welcome, Lead Researcher Mina. The ProtiGen AI is synchronized and ready for analysis.")
    
    # Hero Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h3>343</h3><p>Total Cohort</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h3>97.1%</h3><p>AI Precision</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h3>30</h3><p>Biomarkers</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h3>RTX 3060</h3><p>GPU Active</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("System Overview")
    st.image("https://images.unsplash.com/photo-1576086213369-97a306d36557?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_column_width=True)
    
    st.write("### Quick Access")
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("🚀 Analyze Existing Patient"): 
            st.session_state.current_page = "Analytics"
            st.rerun()
    with btn_c2:
        if st.button("📝 Register New Case"): 
            st.session_state.current_page = "New"
            st.rerun()

# ==========================================
# 5. PAGE: MOLECULAR ANALYTICS (LOOKUP)
# ==========================================
elif st.session_state.current_page == "Analytics":
    st.title("🔬 Advanced Molecular Patient Analytics")
    
    # السحب الذكي: هنسحب بس المرضى اللي ليهم بيانات في الجدولين عشان م يحصلش Error
    conn = sqlite3.connect('ProtiGen_DB.sqlite')
    valid_query = "SELECT DISTINCT m.Patient_ID FROM Microbiome_Profiles m JOIN Patients_Clinical p ON m.Patient_ID = p.Patient_ID"
    valid_patients = pd.read_sql_query(valid_query, conn)['Patient_ID'].tolist()
    
    selected_id = st.selectbox("Search Patient ID in Clinical Database:", valid_patients)
    
    if st.button("RUN DEEP ANALYTICS REPORT", type="primary"):
        with st.spinner("Decoding DNA & Microbiome Patterns..."):
            # سحب الداتا كاملة
            query = f"SELECT m.*, p.* FROM Microbiome_Profiles m JOIN Patients_Clinical p ON m.Patient_ID = p.Patient_ID WHERE m.Patient_ID = '{selected_id}'"
            p_data = pd.read_sql_query(query, conn)
            
            if p_data.empty:
                st.error("❌ Data Inconsistency: Selected ID has no microbiome profile.")
            else:
                # تجهيز البيانات للموديل
                # نأخذ أول صف فقط في حالة التكرار
                single_patient = p_data.iloc[[0]]
                bacteria_features = single_patient[bacteria_list].fillna(0)
                
                # Prediction
                scaled_x = scaler.transform(bacteria_features.values)
                with torch.no_grad():
                    prob = model(torch.tensor(scaled_x, dtype=torch.float32)).item()
                
                is_responder = 1 if prob > 0.5 else 0
                conf_score = prob*100 if is_responder==1 else (1-prob)*100
                
                st.markdown('<div class="report-header"><div class="report-content">', unsafe_allow_html=True)
                st.header(f"Clinical Report: {selected_id}")
                
                # --- القسم الأول: التشخيص الذكي ---
                t1, t2, t3 = st.tabs(["🤖 AI Diagnosis", "📊 Microbial Profile", "💊 Treatment Plan"])
                
                with t1:
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=conf_score,
                            title={'text': "AI Confidence %"},
                            gauge={'bar': {'color': "#238636" if is_responder else "#da3633"},
                                   'axis': {'range': [0, 100]}}
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    with col_b:
                        st.write("### Diagnostic Result")
                        if is_responder:
                            st.success("## TARGET: RESPONDER")
                            st.write("Patient microbiome indicates high susceptibility to current therapy.")
                        else:
                            st.error("## TARGET: NON-RESPONDER")
                            st.write("Warning: High dysbiosis detected. Low probability of therapy success.")
                
                with t2:
                    st.write("### Microbiome Visualization Suite")
                    v1, v2 = st.columns(2)
                    with v1:
                        # Radar Chart
                        radar_vals = bacteria_features.iloc[0][:10].values
                        radar_names = bacteria_features.columns[:10]
                        fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=radar_names, fill='toself'))
                        fig_radar.update_layout(title="Bacterial Fingerprint (Top 10)")
                        st.plotly_chart(fig_radar, use_container_width=True)
                    with v2:
                        # Pie Chart
                        pie_vals = bacteria_features.iloc[0][:8].values
                        pie_names = bacteria_features.columns[:8]
                        st.plotly_chart(px.pie(values=pie_vals, names=pie_names, title="Relative Abundance"), use_container_width=True)
                    
                    # Heatmap
                    st.plotly_chart(px.imshow([bacteria_features.values[0]], x=bacteria_list, title="Full Microbiome Heatmap", color_continuous_scale="Viridis"), use_container_width=True)

                with t3:
                    st.subheader(" Personalized Medical Recommendations")
                    if is_responder:
                        st.info("✅ **Protocol:** Proceed with Standard Immunotherapy.")
                        st.write("✅ **Supportive Care:** Fiber-rich prebiotic diet recommended.")
                    else:
                        st.warning("⚠️ **Protocol:** HALT Standard Immunotherapy.")
                        st.error("⚠️ **Alternative:** Evaluate Fecal Microbiota Transplant (FMT) or TLR9 Agonists.")
                
                st.markdown('</div></div>', unsafe_allow_html=True)
    conn.close()

# ==========================================
# 6. PAGE: NEW PATIENT REGISTRATION
# ==========================================
elif st.session_state.current_page == "New":
    st.title("➕ New Patient Molecular Registration")
    st.write("Enter sequencing values for all 30 biomarkers to generate a report.")
    
    with st.form("precision_form"):
        st.subheader("Microbiome Sequencing Input (Relative Abundance %)")
        new_inputs = {}
        cols = st.columns(4)
        
        # عرض الـ 30 بكتيريا بالظبط عشان السكايلر ميزعلش
        for i, b_name in enumerate(bacteria_list):
            with cols[i % 4]:
                new_inputs[b_name] = st.number_input(f"{b_name[:15]}", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
        
        register_btn = st.form_submit_button("🚀 START AI MOLECULAR ANALYSIS")
        
        if register_btn:
            progress = st.progress(0)
            status = st.empty()
            for p in range(0, 101, 25):
                time.sleep(0.2)
                progress.progress(p)
                status.text(f"Analyzing patterns... {p}%")
            
            # تجميع البيانات بالترتيب الصحيح
            input_vector = [new_inputs[name] for name in bacteria_list]
            
            scaled_vec = scaler.transform([input_vector])
            with torch.no_grad():
                prob_new = model(torch.tensor(scaled_vec, dtype=torch.float32)).item()
            
            res_new = 1 if prob_new > 0.5 else 0
            conf_new = prob_new*100 if res_new==1 else (1-prob_new)*100
            
            st.markdown("---")
            if res_new:
                st.balloons()
                st.success(f"### DIAGNOSIS: RESPONDER (Confidence: {conf_new:.2f}%)")
            else:
                st.error(f"### DIAGNOSIS: NON-RESPONDER (Confidence: {conf_new:.2f}%)")
            
            st.info("Clinical report generated successfully. Ready for printing.")