import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 加载模型（这里假设模型已经训练好）
@st.cache_resource
def load_model():
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    return model

model = load_model()

st.title("HLH 预测模型")

st.sidebar.header("📊 请输入患者数据")
Ferritin = st.sidebar.number_input("Ferritin (ng/mL)", min_value=0)
LDH = st.sidebar.number_input("LDH (IU/L)", min_value=0)
TRIG = st.sidebar.number_input("TRIG (mg/dL)", min_value=0)
TBA = st.sidebar.number_input("TBA (umol/L)", min_value=0)
eGFR = st.sidebar.number_input("eGFR-EPI (mL/min/1.73m²)", min_value=0)

# 🎯 **转换成模型输入格式**
input_data = np.array([[Ferritin, LDH, TRIG, TBA, eGFR]])
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# 🎯 **预测 HLH 风险**
if st.button("🔍 预测 HLH 风险"):
    hlh_probability = model.predict_proba(input_data_scaled)[:, 1]  # 取HLH的概率

    # 设定风险等级
    if hlh_probability[0] > 0.8:
        risk_level = "⚠️ 高风险"
    elif hlh_probability[0] > 0.5:
        risk_level = "⚠️ 中等风险"
    else:
        risk_level = "✅ 低风险"

    # 🎯 **显示预测结果**
    st.write(f"💡 **HLH 预测概率: {hlh_probability[0]:.2f}**")
    st.write(f"🩺 **风险等级: {risk_level}**")

    # 🎯 **下载预测结果**
    result_df = pd.DataFrame({
        "Ferritin": [Ferritin],
        "LDH": [LDH],
        "TRIG": [TRIG],
        "TBA": [TBA],
        "eGFR-EPI": [eGFR],
        "HLH 预测概率": [hlh_probability[0]],
        "风险等级": [risk_level]
    })

    st.download_button(
        label="📥 下载预测结果",
        data=result_df.to_csv(index=False, encoding="utf-8"),
        file_name="HLH_prediction_results.csv",
        mime="text/csv"
    )
