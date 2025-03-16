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

# 用户输入数据
st.sidebar.header("输入患者数据")
LDH = st.sidebar.number_input("LDH (IU/L)", min_value=0)
TRIG = st.sidebar.number_input("Triglycerides (mg/dL)", min_value=0)
Ferritin = st.sidebar.number_input("Ferritin (ng/mL)", min_value=0)
Fib = st.sidebar.number_input("Fibrinogen (g/L)", min_value=0)
WBC = st.sidebar.number_input("WBC (10^9/L)", min_value=0)

# 转换成模型输入格式
input_data = np.array([[LDH, TRIG, Ferritin, Fib, WBC]])
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# 预测
if st.button("预测 HLH 风险"):
    probability = model.predict_proba(input_data_scaled)[:, 1]
    st.write(f"HLH 预测概率: {probability[0]:.2f}")
