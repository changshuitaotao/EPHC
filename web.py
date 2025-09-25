import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib as mpl
from matplotlib import pyplot as plt
from sksurv.ensemble import RandomSurvivalForest

# 设置页面
st.set_page_config(
    page_title="Survival Prediction Model",
    page_icon="📊",
    layout="wide"
)

# 应用自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .chart-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">Random Survival Forest Prediction Model</h1>', unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model():
    # 这里替换为你的模型加载代码
    with open('rsf.pkl', 'rb') as f:
        model = pickle.load(f)
    # 返回一个示例模型（实际使用时请替换为你的模型）
    return model

model = load_model()

# 计算生存率函数
def get_surv(estimator,data,times=np.arange(12, 121,12)):
    """获取任何一个时间点的生存率，适用于scikit-survival包

    Parameters
    ----------
    estimator : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    times : TYPE, optional
        DESCRIPTION. The default is np.arange(12, 121,12).

    Returns
    -------
    surv_arr : TYPE
        DESCRIPTION.

    """
    sf = estimator.predict_survival_function(data,return_array=False)
    surv_arr = np.array([sample(times) for sample in sf])
    return surv_arr

# 输入表单
with st.sidebar:
    st.header("Patient Information Input")
    st.markdown("Please enter clinical features for survival prediction")
    
    # 这里添加你的特征输入字段
    # age = st.slider("年龄", 20, 100, 62)
    # gender = st.selectbox("性别", ("男性", "女性"), index=1)
    # tumor_size = st.slider("肿瘤大小 (cm)", 0.1, 10.0, 3.5)
    # nodes = st.slider("阳性淋巴结数量", 0, 20, 2)
    # grade = st.selectbox("肿瘤分级", (1, 2, 3), index=1)
    # er_status = st.selectbox("ER状态", ("阴性", "阳性"), index=1)
    # 连续变量
    ast_input = st.text_input('AST(g/L)','80')
    crp_input = st.text_input('CRP(mg/L)','20')
    tace_input = st.text_input('TACE sessions','2')
    
    # 分类变量
    bmi_input = st.selectbox('BMI',('Normal weight', 'Overweight', 'Obesity'))
    TKI_PD1_input = st.selectbox('TKI-PD1',('No', 'Yes'))
    OR_input = st.selectbox('OR',('No', 'Yes'))
    Intratumoral_artery_input = st.selectbox('Intratumoral artery',('No', 'Yes'))
    Antiviral_therapy_input = st.selectbox('Antiviral therapy',('No', 'Yes'))
    Vascular_invasion_input = st.selectbox('Vascular invasion',('Absense', 'Presense'))
    Local_therapy_input = st.selectbox('Local therapy',('Absense', 'Presense'))
    
    # 转换输入为模型需要的格式
    ast = (float(ast_input)-77.93601974)/82.77634102507558
    crp = (float(crp_input)-21.14039474)/26.731608031248104
    tace = (float(tace_input)-2.19572368)/1.908993983466879
    bmi_1 = np.where(bmi_input=='Overweight',1,0)
    bmi_2 = np.where(bmi_input=='Obesity',1,0)
    TKI_PD1 = np.where(TKI_PD1_input=='No',0,1)
    OR = np.where(OR_input=='No',0,1)
    Intratumoral_artery = np.where(Intratumoral_artery_input=='No',0,1)
    Antiviral_therapy = np.where(Antiviral_therapy_input=='No',0,1)
    Vascular_invasion = np.where(Vascular_invasion_input=='Absense',0,1)
    Local_therapy = np.where(Local_therapy_input=='Absense',0,1)
    
    # 创建特征 DataFrame
    features = np.array([ast,crp,tace,bmi_1,bmi_2,TKI_PD1,OR,Intratumoral_artery,Antiviral_therapy,Vascular_invasion,Local_therapy]).reshape(1,-1)

#st.title('Postoperative prolonged hospital stay calculator for patients undergoing gastrointestinal surgery')
#st.subheader('Estimates the risk of postoperative prolonged hospital stay following gastrointestinal surgery in patients.')
#st.divider()
#st.header('Pearls/Pitfalls')
#st.write('This calculator is a data-driven risk stratification tool designed to estimate the likelihood of prolonged hospital stay after gastrointestinal surgery in patients. It is based on the FDP-PONV randomized controlled trial which included 1141 patients and integrates extreme gradient boosting algorithm to help identify high-risk individuals and optimize postoperative management.')
#st.header('When to use')
#st.write('Patients undergoing gastrointestinal surgery, to estimate the risk of postoperative prolonged hospital stay.')
#st.header('Why use')
#st.write('Early identification of patients at high risk for prolonged hospitalization enables clinicians to anticipate and allocate postoperative care resources more effectively, guide patient and family discussions about recovery expectations, minimize unnecessary hospital days and associated costs, and support personalized interventions and discharge planning. By leveraging comprehensive perioperative data, this tool facilitates objective risk stratification to enhance clinical outcomes, optimize resource utilization, and improve overall patient satisfaction.')
#st.divider()

if st.sidebar.button("Predict Survival Rate", type="primary"):
    col1, col2 = st.columns(2)
    survival_rates = get_surv(model,features).ravel()
    years=np.arange(12, 121,12)
    # 创建生存率表格
    survival_data = []
    for year, rate in zip(years, survival_rates):
        survival_data.append({
            "年份": f"第 {year} 年",
            "生存率": f"{rate:.1%}",
            "风险": f"{(1-rate):.1%}"
        })
    
    survival_df = pd.DataFrame(survival_data)
    # 显示预测结果
    st.markdown(f'<div class="prediction-card"><h3>Prediction Results</h3><p>Based on the input features, the 10-year survival prediction is as follows：</p></div>', unsafe_allow_html=True)
    
    # 创建生存曲线图容器
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # 创建生存曲线图
    st.subheader("📊 Survival Curve")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 设置美观的样式
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    
    # 绘制生存曲线
    print(survival_rates*100)
    ax.plot(years, survival_rates*100, linewidth=3, color='#2E86AB', marker='o', markersize=8)
    ax.fill_between(years, survival_rates*100, alpha=0.2, color='#2E86AB')
    
    # 设置标题和标签
    ax.set_title('', fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel('Survival time in months', fontsize=14, fontweight='bold')
    ax.set_ylabel(' Survival probability', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for i, rate in enumerate(survival_rates):
        ax.annotate(f'{rate:.2%}', (years[i], rate*100), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=11, fontweight='bold')
    
    # 美化图表
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.5)
    ax.spines['bottom'].set_alpha(0.5)
    
    # 添加水平参考线
    for y in np.arange(0.1, 1.0, 0.1):
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
    
    # 设置x轴刻度
    ax.set_xticks(years)
    
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 添加关键指标
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("1-Year Survival", f"{survival_rates[0]:.2%}")
    with col2:
        st.metric("5-Year Survival", f"{survival_rates[4]:.2%}")
    with col3:
        st.metric("10-Year Survival", f"{survival_rates[9]:.2%}")
    
    # 添加解释性文本
    #st.info("""
    #**说明**: 
    #- 生存率表示从当前时间点开始，患者存活到各时间点的概率
    #- 该预测基于随机生存森林模型，使用类似患者的历史数据训练
    #- 实际结果可能因个体差异而不同，请结合临床判断使用
    #""")
else:
    st.info("Please enter patient information on the left and click 'Predict Survival Rate' to view results.")

# 添加页脚
st.markdown("---")
st.markdown("© 2025 Chinese PLA General Hospital| Based on Random Survival Forest Model")