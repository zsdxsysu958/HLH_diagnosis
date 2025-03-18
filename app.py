import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 🔗 GitHub RAW 链接（替换为你的仓库）
MODEL_URL = "https://raw.githubusercontent.com/zsdxsysu958/HLH_diagnosis/main/random_forest_model.pkl"
SCALER_URL = "https://raw.githubusercontent.com/zsdxsysu958/HLH_diagnosis/main/scaler.pkl"

# 🎯 **加载模型**
@st.cache_resource
def load_model():
    model_path = "/mnt/data/random_forest_model.pkl"
    scaler_path = "/mnt/data/scaler.pkl"

    # 如果模型文件不存在，则从 GitHub 下载
    if not os.path.exists(model_path):
        st.info("🔽 正在从 GitHub 下载模型文件...")
        response = requests.get(MODEL_URL)
        with open(model_path, "wb") as f:
            f.write(response.content)

    if not os.path.exists(scaler_path):
        st.info("🔽 正在从 GitHub 下载标准化工具...")
        response = requests.get(SCALER_URL)
        with open(scaler_path, "wb") as f:
            f.write(response.content)

    # 加载模型
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# 加载模型
model, scaler = load_model()
st.success("✅ 模型加载成功，可以进行预测！")

# 🎯 **页面标题**
st.title("🩺 HLH 预测模型")

st.sidebar.header("📊 请输入患者数据")
Ferritin = st.sidebar.number_input("Ferritin (ng/mL)", min_value=0)
LDH = st.sidebar.number_input("LDH (IU/L)", min_value=0)
TRIG = st.sidebar.number_input("TRIG (mg/dL)", min_value=0)
TBA = st.sidebar.number_input("TBA (umol/L)", min_value=0)
eGFR = st.sidebar.number_input("eGFR-EPI (mL/min/1.73m²)", min_value=0)

# 🎯 **转换成模型输入格式**
input_data = np.array([[Ferritin, LDH, TRIG, TBA, eGFR]])
input_data_scaled = scaler.transform(input_data)  # 使用训练时的scaler进行标准化

# 🎯 **预测 HLH 风险**
if st.button("🔍 预测 HLH 风险"):
    hlh_probability = model.predict_proba(input_data_scaled)[:, 1]  # 获取 HLH 预测概率

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
