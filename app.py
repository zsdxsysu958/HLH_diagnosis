import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# 🎯 **加载模型**
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")  # 加载训练好的模型
    scaler = joblib.load("scaler.pkl")  # 加载用于数据标准化的Scaler
    return model, scaler

model, scaler = load_model()

# 🎯 **页面标题**
st.title("🩺 噬血细胞综合征（Hemophagocytic Lymphohistiocytosis, HLH） 预测模型")

st.sidebar.header("📊 请输入患者数据")
Ferritin = st.sidebar.number_input("Ferritin (ng/mL)", min_value=0.0, step=0.01)
LDH = st.sidebar.number_input("LDH (IU/L)", min_value=0.0, step=0.01)
TRIG = st.sidebar.number_input("TRIG (mg/dL)", min_value=0.0, step=0.01)
TBA = st.sidebar.number_input("TBA (umol/L)", min_value=0.0, step=0.01)
eGFR = st.sidebar.number_input("eGFR-EPI (mL/min/1.73m²)", min_value=0.0, step=0.01)

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

    import io

# 创建一个 BytesIO 缓存区
output = io.BytesIO()

# 将 DataFrame 写入 Excel
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    result_df.to_excel(writer, index=False, sheet_name="HLH 预测")

# 让 Streamlit 生成 Excel 下载按钮
st.download_button(
    label="📥 下载预测结果 (Excel)",
    data=output.getvalue(),
    file_name="HLH_prediction_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
