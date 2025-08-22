import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
import xgboost as xgb
import time
import uuid
from io import BytesIO, StringIO
from PIL import Image
import io
import base64
import joblib
import os
from tempfile import NamedTemporaryFile

# 设置中文字体支持（解决乱码问题）
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimSun", "SimHei", "KaiTi", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
sns.set(font="Microsoft YaHei")  # 为seaborn单独设置字体

# 页面配置
st.set_page_config(
    page_title="机器学习可视化平台",
    page_icon="📊",
    layout="wide"
)

# 确保临时目录存在
if not os.path.exists("temp"):
    os.makedirs("temp")

# 初始化会话状态
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    
    if "dataset_type" not in st.session_state:
        st.session_state.dataset_type = "built_in"  # built_in 或 custom
    
    if "task_type" not in st.session_state:
        st.session_state.task_type = "classification"  # classification 或 regression
    
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    
    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    
    if "model" not in st.session_state:
        st.session_state.model = None
    
    if "model_name" not in st.session_state:
        st.session_state.model_name = "未选择模型"
    
    if "train_metrics" not in st.session_state:
        st.session_state.train_metrics = {}
    
    if "test_metrics" not in st.session_state:
        st.session_state.test_metrics = {}
    
    if "is_trained" not in st.session_state:
        st.session_state.is_trained = False
    
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    
    if "y_pred" not in st.session_state:
        st.session_state.y_pred = None
    
    if "training_time" not in st.session_state:
        st.session_state.training_time = 0.0
    
    if "feature_names" not in st.session_state:
        st.session_state.feature_names = None
    
    if "target_names" not in st.session_state:
        st.session_state.target_names = None
    
    # 存储原始标签编码器，用于后续映射
    if "label_encoder" not in st.session_state:
        st.session_state.label_encoder = None
    
    # 存储历史训练记录
    if "training_history" not in st.session_state:
        st.session_state.training_history = []
    
    # 特征工程相关
    if "categorical_cols" not in st.session_state:
        st.session_state.categorical_cols = []
    
    if "numerical_cols" not in st.session_state:
        st.session_state.numerical_cols = []

# 加载内置数据集
@st.cache_data
def load_builtin_dataset(dataset_name, task_type):
    # 存储数据集元信息，用于后续识别
    dataset_info = {
        "classification": {
            "鸢尾花数据集": {"loader": datasets.load_iris, "name_key": "iris"},
            "乳腺癌数据集": {"loader": datasets.load_breast_cancer, "name_key": "breast_cancer"},
            "葡萄酒数据集": {"loader": datasets.load_wine, "name_key": "wine"}
        },
        "regression": {
            "糖尿病数据集": {"loader": datasets.load_diabetes, "name_key": "diabetes"},
            "加州房价数据集": {"loader": datasets.fetch_california_housing, "name_key": "california_housing"}
        }
    }
    
    if task_type not in dataset_info or dataset_name not in dataset_info[task_type]:
        return None, None, None, None, None, None, None  # 返回完整None元组
    
    try:
        # 加载数据集
        data = dataset_info[task_type][dataset_name]["loader"]()
        # 存储数据集标识，用于后续识别
        data["name_key"] = dataset_info[task_type][dataset_name]["name_key"]
        
        # 特殊处理加州房价数据集，它的data和target属性名称不同
        if data["name_key"] == "california_housing":
            X = data.data
            y = data.target
        else:
            X = data.data
            y = data.target
        
        # 处理特征名称
        if hasattr(data, 'feature_names'):
            feature_names = data.feature_names
        else:
            feature_names = [f"特征{i+1}" for i in range(X.shape[1])]
        
        # 处理目标名称
        target_names = None
        if task_type == "classification" and hasattr(data, 'target_names'):
            target_names = data.target_names
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, feature_names, target_names, data
    except Exception as e:
        st.error(f"加载数据集失败: {str(e)}")
        return None, None, None, None, None, None, None

# 加载用户上传的数据集
@st.cache_data
def load_custom_dataset(file, file_type):
    try:
        if file_type == "csv":
            # 限制文件大小，防止过大文件
            if file.size > 200 * 1024 * 1024:  # 200MB
                return None, "文件过大，请上传小于200MB的文件"
            
            # 对于大型CSV，先读取部分行进行预览
            if file.size > 10 * 1024 * 1024:  # 10MB
                file.seek(0)
                df = pd.read_csv(file, nrows=10000)  # 只读取10000行
                return df, "成功加载数据预览（仅前10000行）"
            else:
                file.seek(0)
                df = pd.read_csv(file)
        elif file_type in ["xls", "xlsx"]:
            file.seek(0)
            df = pd.read_excel(file)
        else:
            return None, "不支持的文件类型"
        
        return df, "成功加载数据"
    except Exception as e:
        return None, f"加载失败: {str(e)}"

# 数据清洗处理
def clean_data(df, missing_value_strategy, categorical_encoding, target_col=None, task_type="classification"):
    # 复制数据避免修改原始数据
    cleaned_df = df.copy()
    
    # 识别数值型和类别型特征
    numerical_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 如果指定了目标列，从特征列表中移除
    if target_col and target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # 处理缺失值
    if missing_value_strategy == "删除行":
        cleaned_df = cleaned_df.dropna()
    elif missing_value_strategy == "均值填充":
        # 对数值特征使用均值填充
        cleaned_df[numerical_cols] = cleaned_df[numerical_cols].fillna(cleaned_df[numerical_cols].mean())
        # 对类别特征使用众数填充
        for col in categorical_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else "")
    elif missing_value_strategy == "中位数填充":
        # 对数值特征使用中位数填充
        cleaned_df[numerical_cols] = cleaned_df[numerical_cols].fillna(cleaned_df[numerical_cols].median())
        # 对类别特征使用众数填充
        for col in categorical_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else "")
    
    # 存储标签编码器和原始类别名称
    label_encoder = None
    target_names = None
    
    # 如果指定了目标列，处理目标列的编码（仅分类任务）
    if task_type == "classification" and target_col and target_col in cleaned_df.columns:
        # 检查目标列是否为类别型
        if cleaned_df[target_col].dtype in ['object', 'category']:
            # 对目标列进行编码
            label_encoder = LabelEncoder()
            cleaned_df[target_col] = label_encoder.fit_transform(cleaned_df[target_col])
            target_names = label_encoder.classes_  # 原始类别名称
    
    # 对其他类别特征进行编码
    for col in categorical_cols:
        if col != target_col:  # 已处理过目标列
            if categorical_encoding == "标签编码":
                le = LabelEncoder()
                cleaned_df[col] = le.fit_transform(cleaned_df[col].astype(str))
            elif categorical_encoding == "独热编码":
                # 使用独热编码
                dummies = pd.get_dummies(cleaned_df[col], prefix=col, drop_first=True)
                cleaned_df = pd.concat([cleaned_df, dummies], axis=1)
                cleaned_df.drop(col, axis=1, inplace=True)
    
    return cleaned_df, target_names, label_encoder, numerical_cols, categorical_cols

# 准备训练数据
def prepare_training_data(df, target_col, test_size=0.3, task_type="classification"):
    if target_col not in df.columns:
        return None, None, None, None, None, None, "目标列不存在"
    
    # 分离特征和目标
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    # 获取处理后数据中实际存在的类别（仅分类任务）
    unique_classes = None
    if task_type == "classification":
        unique_classes = np.unique(y)
    
    return X_train, X_test, y_train, y_test, feature_names, unique_classes, "数据准备完成"

# 创建模型
def create_model(model_name, params, task_type="classification"):
    try:
        if task_type == "classification":
            if model_name == "K近邻分类器":
                return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
            elif model_name == "支持向量机":
                return SVC(C=params["C"], kernel=params["kernel"], probability=True, random_state=42)
            elif model_name == "决策树":
                return DecisionTreeClassifier(max_depth=params["max_depth"], random_state=42)
            elif model_name == "随机森林":
                return RandomForestClassifier(n_estimators=params["n_estimators"], random_state=42)
            elif model_name == "逻辑回归":
                return LogisticRegression(C=params["C"], max_iter=1000, random_state=42)
            elif model_name == "梯度提升树":
                return GradientBoostingClassifier(n_estimators=params["n_estimators"], random_state=42)
            elif model_name == "XGBoost":
                return xgb.XGBClassifier(n_estimators=params["n_estimators"], random_state=42)
        
        elif task_type == "regression":
            if model_name == "K近邻回归器":
                return KNeighborsRegressor(n_neighbors=params["n_neighbors"])
            elif model_name == "支持向量回归":
                return SVR(C=params["C"], kernel=params["kernel"])
            elif model_name == "决策树回归":
                return DecisionTreeRegressor(max_depth=params["max_depth"], random_state=42)
            elif model_name == "随机森林回归":
                return RandomForestRegressor(n_estimators=params["n_estimators"], random_state=42)
            elif model_name == "线性回归":
                return LinearRegression()
            elif model_name == "岭回归":
                return Ridge(alpha=params["alpha"])
            elif model_name == "Lasso回归":
                return Lasso(alpha=params["alpha"], max_iter=1000)
            elif model_name == "梯度提升回归":
                return GradientBoostingRegressor(n_estimators=params["n_estimators"], random_state=42)
            elif model_name == "XGBoost回归":
                return xgb.XGBRegressor(n_estimators=params["n_estimators"], random_state=42)
        
        return None
    except Exception as e:
        st.error(f"创建模型失败: {str(e)}")
        return None

# 超参数调优
def tune_model(model, param_grid, X_train, y_train, cv=3):
    try:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy' if st.session_state.task_type == "classification" else 'neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 自定义训练循环以显示进度
        for i in range(1, 101):
            time.sleep(0.01)  # 模拟进度
            progress_bar.progress(i)
            status_text.text(f"超参数搜索进度: {i}%")
            
            # 实际训练只执行一次
            if i == 1:
                grid_search.fit(X_train, y_train)
        
        status_text.text("超参数搜索完成!")
        return grid_search.best_estimator_, grid_search.best_params_
    except Exception as e:
        st.error(f"超参数调优失败: {str(e)}")
        return model, {}

# 训练模型
def train_model(model, X_train, y_train):
    try:
        start_time = time.time()
        
        # 添加进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 自定义训练循环以显示进度（实际训练只执行一次）
        for i in range(1, 101):
            time.sleep(0.01)  # 模拟进度
            progress_bar.progress(i)
            status_text.text(f"模型训练进度: {i}%")
            
            # 实际训练只执行一次
            if i == 1:
                model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        status_text.text(f"训练完成! 耗时: {training_time:.4f}秒")
        return model, training_time, None
    except ValueError as e:
        return None, 0, f"数据格式错误: {str(e)}，请检查特征是否包含非数值类型"
    except Exception as e:
        return None, 0, f"训练失败: {str(e)}"

# 评估模型
def evaluate_model(model, X, y, task_type="classification", target_names=None):
    try:
        y_pred = model.predict(X)
        
        if task_type == "classification":
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }, y_pred
        else:  # regression
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }, y_pred
    except Exception as e:
        st.error(f"模型评估失败: {str(e)}")
        return {}, None

# 保存模型
def save_model(model, model_name):
    try:
        with NamedTemporaryFile(delete=False, suffix='.pkl', dir='temp') as tmp_file:
            joblib.dump(model, tmp_file)
            return tmp_file.name
    except Exception as e:
        st.error(f"保存模型失败: {str(e)}")
        return None

# 加载模型
def load_model(file):
    try:
        return joblib.load(file)
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None

# 绘制混淆矩阵（交互式）
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm, 
        labels=dict(x="预测标签", y="真实标签", color="数量"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        title="混淆矩阵",
        width=700,
        height=600,
    )
    return fig

# 绘制学习曲线（交互式）
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, 
        scoring='accuracy' if st.session_state.task_type == "classification" else 'neg_mean_squared_error',
        n_jobs=-1, 
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # 对于回归任务，转换为正数以便展示
    if st.session_state.task_type == "regression":
        train_mean = -train_mean
        test_mean = -test_mean
    
    fig = go.Figure()
    
    # 添加训练准确率曲线
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='训练分数',
        line=dict(color='blue'),
        marker=dict(symbol='circle')
    ))
    
    # 添加训练准确率误差带
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean + train_std,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean - train_std,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.1)',
        line=dict(width=0),
        showlegend=False
    ))
    
    # 添加验证准确率曲线
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        mode='lines+markers',
        name='验证分数',
        line=dict(color='green'),
        marker=dict(symbol='square')
    ))
    
    # 添加验证准确率误差带
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean + test_std,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean - test_std,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='训练样本数',
        yaxis_title='准确率' if st.session_state.task_type == "classification" else '负均方误差',
        width=800,
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

# 绘制特征重要性（交互式）
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 只显示前15个特征，避免太多特征导致图表拥挤
        if len(importances) > 15:
            indices = indices[:15]
            importances = importances[:15]
            feature_names = [feature_names[i] for i in indices]
        
        fig = px.bar(
            x=feature_names,
            y=importances,
            labels={'x': '特征', 'y': '重要性'},
            title='特征重要性',
            color=importances,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            width=800,
            height=500,
            xaxis_tickangle=-45
        )
        return fig
    return None

# 绘制ROC曲线（交互式）
def plot_roc_curve(model, X_test, y_test, class_names):
    if len(np.unique(y_test)) == 2:  # 二分类问题
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # 添加ROC曲线
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC曲线 (面积 = {roc_auc:.2f})',
            line=dict(color='darkorange', width=2)
        ))
        
        # 添加对角线
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='随机猜测',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC曲线',
            xaxis_title='假正例率',
            yaxis_title='真正例率',
            xaxis=dict(range=[0.0, 1.0]),
            yaxis=dict(range=[0.0, 1.05]),
            width=800,
            height=500,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    return None

# 绘制预测值与真实值对比图（回归任务）
def plot_prediction_vs_actual(y_true, y_pred):
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={'x': '真实值', 'y': '预测值'},
        title='预测值 vs 真实值'
    )
    
    # 添加理想线（y=x）
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='理想预测',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        width=800,
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    return fig

# 绘制残差图（回归任务）
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={'x': '预测值', 'y': '残差 (真实值-预测值)'},
        title='残差图'
    )
    
    # 添加水平线y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        width=800,
        height=500
    )
    return fig

# 数据集下载功能
def get_dataset_download_link(df, filename, text):
    """生成数据集下载链接"""
    try:
        # 根据文件类型转换数据
        if filename.endswith('.csv'):
            # 转换为CSV字符串
            csv = df.to_csv(index=False)
            b = csv.encode()
        elif filename.endswith('.xlsx'):
            # 转换为Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='数据集')
            b = output.getvalue()
        
        # 生成下载链接
        href = f'<a href="data:file/csv;base64,{base64.b64encode(b).decode()}" download="{filename}">{text}</a>'
        return href
    except Exception as e:
        st.error(f"生成下载链接失败: {str(e)}")
        return None

# 主函数
def main():
    # 初始化会话状态
    init_session_state()
    
    # 页面标题
    st.title("📊 机器学习可视化平台")
    st.markdown("---")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("🔧 配置面板")
        
        # 会话信息
        st.subheader("会话信息")
        st.write(f"**会话ID:** `{st.session_state.session_id[:8]}...`")
        
        st.markdown("---")
        
        # 任务类型选择
        st.subheader("1. 任务类型")
        task_type = st.radio(
            "选择机器学习任务类型",
            ["分类任务", "回归任务"],
            index=0 if st.session_state.task_type == "classification" else 1
        )
        
        # 更新任务类型状态
        st.session_state.task_type = "classification" if task_type == "分类任务" else "regression"
        
        st.markdown("---")
        
        # 数据集来源选择
        st.subheader("2. 数据集来源")
        dataset_source = st.radio(
            "选择数据集类型",
            ["内置数据集", "自定义数据集上传"],
            index=0 if st.session_state.dataset_type == "built_in" else 1
        )
        
        # 更新数据集类型状态
        if dataset_source == "内置数据集":
            st.session_state.dataset_type = "built_in"
        else:
            st.session_state.dataset_type = "custom"
        
        # 内置数据集选择
        if st.session_state.dataset_type == "built_in":
            dataset_options = ["鸢尾花数据集", "乳腺癌数据集", "葡萄酒数据集"] if st.session_state.task_type == "classification" else \
                             ["糖尿病数据集", "加州房价数据集"]
            
            dataset_name = st.selectbox(
                "选择示例数据集",
                dataset_options
            )
            
            # 加载数据集按钮
            if st.button("加载数据集", use_container_width=True):
                with st.spinner("正在加载数据集..."):
                    result = load_builtin_dataset(dataset_name, st.session_state.task_type)
                    if result[0] is not None:
                        X_train, X_test, y_train, y_test, feature_names, target_names, data = result
                        st.session_state.dataset = data
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = feature_names
                        st.session_state.target_names = target_names
                        st.success(f"✅ 已成功加载 {dataset_name}")
                        st.session_state.is_trained = False  # 重新加载数据集后重置模型状态
        
        # 自定义数据集上传
        else:
            st.subheader("上传数据集")
            uploaded_file = st.file_uploader("上传CSV或Excel文件", type=["csv", "xls", "xlsx"])
            
            # 数据加载
            if uploaded_file is not None:
                file_type = uploaded_file.name.split(".")[-1].lower()
                with st.spinner("正在加载数据..."):
                    df, message = load_custom_dataset(uploaded_file, file_type)
                    if df is not None:
                        st.session_state.raw_data = df
                        st.success(f"✅ {message}")
                        st.info(f"数据集包含 {df.shape[0]} 行, {df.shape[1]} 列")
                        
                        # 显示数据预览
                        with st.expander("查看数据预览"):
                            st.dataframe(df.head(10), use_container_width=True)
            
            # 数据清洗设置
            if st.session_state.raw_data is not None:
                st.subheader("3. 数据清洗设置")
                
                # 显示缺失值情况
                missing_values = st.session_state.raw_data.isnull().sum()
                missing_values = missing_values[missing_values > 0]
                if not missing_values.empty:
                    st.warning("检测到缺失值:")
                    st.write(missing_values)
                
                # 缺失值处理策略
                missing_strategy = st.selectbox(
                    "缺失值处理方式",
                    ["删除行", "均值填充", "中位数填充"]
                )
                
                # 类别特征编码方式
                st.subheader("4. 特征编码设置")
                categorical_encoding = st.selectbox(
                    "类别特征编码方式",
                    ["标签编码", "独热编码"]
                )
                
                # 选择目标列（标签列）
                st.subheader("5. 选择目标列")
                target_col = st.selectbox(
                    "选择作为预测目标的列",
                    st.session_state.raw_data.columns.tolist()
                )
                
                # 划分比例
                test_size = st.slider(
                    "测试集比例",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.3,
                    step=0.1
                )
                
                # 准备数据按钮
                if st.button("准备训练数据", use_container_width=True):
                    with st.spinner("正在处理数据..."):
                        # 清洗数据
                        cleaned_df, all_target_names, label_encoder, numerical_cols, categorical_cols = clean_data(
                            st.session_state.raw_data, 
                            missing_strategy,
                            categorical_encoding,
                            target_col,
                            st.session_state.task_type
                        )
                        
                        # 准备训练数据
                        result = prepare_training_data(cleaned_df, target_col, test_size, st.session_state.task_type)
                        if result[0] is not None:
                            X_train, X_test, y_train, y_test, feature_names, unique_classes, msg = result
                            
                            # 更新状态
                            st.session_state.processed_data = cleaned_df
                            st.session_state.target_column = target_col
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.feature_names = feature_names
                            st.session_state.label_encoder = label_encoder  # 保存标签编码器
                            st.session_state.numerical_cols = numerical_cols
                            st.session_state.categorical_cols = categorical_cols
                            
                            # 设置目标名称
                            if st.session_state.task_type == "classification":
                                if all_target_names is not None and label_encoder is not None:
                                    # 映射实际存在的类别到原始名称
                                    st.session_state.target_names = [all_target_names[i] for i in unique_classes]
                                else:
                                    st.session_state.target_names = [f"类别 {i}" for i in unique_classes]
                            else:
                                st.session_state.target_names = ["目标值"]
                            
                            st.success(f"✅ 数据准备完成: {msg}")
                            st.write(f"训练集样本数: {X_train.shape[0]}")
                            st.write(f"测试集样本数: {X_test.shape[0]}")
                            st.write(f"特征数: {X_train.shape[1]}")
                            if st.session_state.task_type == "classification":
                                st.write(f"实际类别数: {len(unique_classes)}")  # 显示实际类别数
                            st.session_state.is_trained = False  # 重新准备数据后重置模型状态
        
        # 检查数据集是否已加载
        if (st.session_state.dataset is not None and st.session_state.dataset_type == "built_in") or \
           (st.session_state.processed_data is not None and st.session_state.dataset_type == "custom"):
            if st.session_state.dataset_type == "built_in":
                # 使用更可靠的方式识别数据集名称
                name_key = st.session_state.dataset.get("name_key")
                dataset_name_map = {
                    "iris": "鸢尾花数据集",
                    "breast_cancer": "乳腺癌数据集",
                    "wine": "葡萄酒数据集",
                    "diabetes": "糖尿病数据集",
                    "california_housing": "加州房价数据集"
                }
                dataset_name = dataset_name_map.get(name_key, "未知数据集")
                st.info(f"已加载: {dataset_name}")
            else:
                st.info(f"已加载: 自定义数据集")
            
            # 只有当X_train存在时才显示这些信息
            if st.session_state.X_train is not None:
                st.write(f"特征数: {st.session_state.X_train.shape[1]}")
                st.write(f"样本数: {st.session_state.X_train.shape[0] + st.session_state.X_test.shape[0]}")
                # 显示实际类别数（仅分类任务）
                if st.session_state.task_type == "classification" and st.session_state.y_train is not None:
                    st.write(f"实际类别数: {len(np.unique(st.session_state.y_train))}")
        
        st.markdown("---")
        
        # 模型选择（仅当数据集已加载时显示）
        if (st.session_state.dataset is not None and st.session_state.dataset_type == "built_in") or \
           (st.session_state.processed_data is not None and st.session_state.dataset_type == "custom"):
            st.subheader("6. 选择模型")
            
            # 根据任务类型显示不同的模型选项
            if st.session_state.task_type == "classification":
                model_name = st.selectbox(
                    "选择分类模型",
                    ["K近邻分类器", "支持向量机", "决策树", "随机森林", "逻辑回归", "梯度提升树", "XGBoost"]
                )
            else:
                model_name = st.selectbox(
                    "选择回归模型",
                    ["K近邻回归器", "支持向量回归", "决策树回归", "随机森林回归", "线性回归", 
                     "岭回归", "Lasso回归", "梯度提升回归", "XGBoost回归"]
                )
            
            # 模型参数配置
            st.subheader("7. 模型参数")
            model_params = {}
            
            if model_name in ["K近邻分类器", "K近邻回归器"]:
                model_params["n_neighbors"] = st.slider("近邻数量", 1, 20, 5)
            
            elif model_name in ["支持向量机", "支持向量回归"]:
                model_params["C"] = st.slider("正则化参数 C", 0.01, 10.0, 1.0)
                model_params["kernel"] = st.selectbox("核函数", ["linear", "poly", "rbf", "sigmoid"])
            
            elif model_name in ["决策树", "决策树回归"]:
                model_params["max_depth"] = st.slider("最大深度", 1, 20, 5)
            
            elif model_name in ["随机森林", "随机森林回归", "梯度提升树", "梯度提升回归", "XGBoost", "XGBoost回归"]:
                model_params["n_estimators"] = st.slider("树的数量", 10, 200, 100)
            
            elif model_name == "逻辑回归":
                model_params["C"] = st.slider("正则化参数 C", 0.01, 10.0, 1.0)
            
            elif model_name in ["岭回归", "Lasso回归"]:
                model_params["alpha"] = st.slider("正则化系数", 0.01, 10.0, 1.0)
            
            # 超参数调优选项
            st.subheader("8. 超参数调优")
            use_grid_search = st.checkbox("启用网格搜索调优超参数", value=False)
            
            st.markdown("---")
            
            # 训练模型按钮
            st.subheader("9. 训练模型")
            if st.button("开始训练", use_container_width=True):
                with st.spinner("正在准备训练..."):
                    # 创建模型
                    model = create_model(model_name, model_params, st.session_state.task_type)
                    if model is not None:
                        # 如果启用了网格搜索，则进行超参数调优
                        if use_grid_search:
                            # 定义简单的参数网格
                            param_grid = {}
                            if model_name in ["K近邻分类器", "K近邻回归器"]:
                                param_grid["n_neighbors"] = [3, 5, 7, 9]
                            elif model_name in ["随机森林", "随机森林回归", "梯度提升树", "梯度提升回归"]:
                                param_grid["n_estimators"] = [50, 100, 150]
                            
                            if param_grid:  # 只有当有参数可优化时才进行网格搜索
                                st.subheader("正在进行超参数调优...")
                                model, best_params = tune_model(model, param_grid, 
                                                              st.session_state.X_train, 
                                                              st.session_state.y_train)
                                st.success(f"最佳参数: {best_params}")
                            else:
                                st.info("该模型暂不支持网格搜索调优，将使用默认参数训练")
                        
                        # 训练模型
                        st.subheader("正在训练模型...")
                        trained_model, training_time, error_msg = train_model(
                            model, 
                            st.session_state.X_train, 
                            st.session_state.y_train
                        )
                        
                        if error_msg:
                            st.error(error_msg)
                        elif trained_model is not None:
                            # 评估模型
                            train_metrics, _ = evaluate_model(
                                trained_model, 
                                st.session_state.X_train, 
                                st.session_state.y_train,
                                st.session_state.task_type,
                                st.session_state.target_names
                            )
                            
                            test_metrics, y_pred = evaluate_model(
                                trained_model, 
                                st.session_state.X_test, 
                                st.session_state.y_test,
                                st.session_state.task_type,
                                st.session_state.target_names
                            )
                            
                            # 更新会话状态
                            st.session_state.model = trained_model
                            st.session_state.model_name = model_name
                            st.session_state.train_metrics = train_metrics
                            st.session_state.test_metrics = test_metrics
                            st.session_state.y_pred = y_pred
                            st.session_state.training_time = training_time
                            st.session_state.is_trained = True
                            
                            # 保存到训练历史
                            history_entry = {
                                "model_name": model_name,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "train_metrics": train_metrics,
                                "test_metrics": test_metrics,
                                "training_time": training_time
                            }
                            st.session_state.training_history.append(history_entry)
                            
                            st.success("✅ 模型训练完成！")
                            if st.session_state.task_type == "classification":
                                st.write(f"训练准确率: {train_metrics.get('accuracy', 0):.4f}")
                                st.write(f"测试准确率: {test_metrics.get('accuracy', 0):.4f}")
                            else:
                                st.write(f"训练R²分数: {train_metrics.get('r2', 0):.4f}")
                                st.write(f"测试R²分数: {test_metrics.get('r2', 0):.4f}")
                            st.write(f"训练时间: {training_time:.4f}秒")
        
        # 模型保存和加载
        if st.session_state.is_trained:
            st.markdown("---")
            st.subheader("10. 模型管理")
            
            # 保存模型
            if st.button("💾 保存当前模型", use_container_width=True):
                model_path = save_model(st.session_state.model, st.session_state.model_name)
                if model_path:
                    with open(model_path, "rb") as file:
                        st.download_button(
                            label="下载模型文件",
                            data=file,
                            file_name=f"{st.session_state.model_name}_{time.strftime('%Y%m%d_%H%M%S')}.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
        
        # 加载已有模型
        st.subheader("11. 加载已有模型")
        uploaded_model = st.file_uploader("上传模型文件 (.pkl)", type=["pkl"])
        if uploaded_model is not None:
            with st.spinner("正在加载模型..."):
                model = load_model(uploaded_model)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_name = f"加载的模型: {uploaded_model.name}"
                    st.session_state.is_trained = True
                    
                    # 评估加载的模型
                    if st.session_state.X_test is not None and st.session_state.y_test is not None:
                        test_metrics, y_pred = evaluate_model(
                            model, 
                            st.session_state.X_test, 
                            st.session_state.y_test,
                            st.session_state.task_type,
                            st.session_state.target_names
                        )
                        st.session_state.test_metrics = test_metrics
                        st.session_state.y_pred = y_pred
                        st.success("✅ 模型加载成功并完成评估！")
        
        st.markdown("---")
        
        # 重置会话按钮
        if st.button("🔄 重置会话", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.success("会话已重置")
            st.rerun()
    
    # 主内容区域
    if (st.session_state.dataset is None and st.session_state.dataset_type == "built_in") or \
       (st.session_state.raw_data is None and st.session_state.dataset_type == "custom"):
        # 初始页面提示
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            ### 欢迎使用机器学习可视化平台
            
            这个平台可以帮助您：
            - 探索内置数据集或上传自己的数据
            - 进行基础数据清洗和预处理
            - 训练多种机器学习模型（分类和回归）
            - 可视化模型性能和结果
            - 比较不同模型的表现
            - 保存和加载训练好的模型
            
            请从侧边栏开始，选择任务类型、数据集类型并加载数据，然后选择模型进行训练。
            """)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1517694712202-14dd9538aa97?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", 
                     caption="机器学习与数据可视化")
    else:
        # 数据集概览与下载
        st.header("📋 数据集概览")
        # 添加下载按钮行
        col_download = st.columns(1)
        with col_download[0]:
            st.subheader("数据集下载")
            if st.session_state.dataset_type == "built_in":
                # 内置数据集下载
                name_key = st.session_state.dataset.get("name_key")
                dataset_name_map = {
                    "iris": "鸢尾花数据集",
                    "breast_cancer": "乳腺癌数据集",
                    "wine": "葡萄酒数据集",
                    "diabetes": "糖尿病数据集",
                    "california_housing": "加州房价数据集"
                }
                dataset_name = dataset_name_map.get(name_key, "未知数据集")
                
                # 创建内置数据集的DataFrame
                if name_key == "california_housing":
                    df = pd.DataFrame(
                        data=st.session_state.dataset.data,
                        columns=st.session_state.dataset.feature_names
                    )
                    df['目标值'] = st.session_state.dataset.target
                else:
                    df = pd.DataFrame(
                        data=st.session_state.dataset.data,
                        columns=st.session_state.dataset.feature_names
                    )
                    if st.session_state.task_type == "classification":
                        df['类别'] = [st.session_state.dataset.target_names[i] for i in st.session_state.dataset.target]
                    else:
                        df['目标值'] = st.session_state.dataset.target
                
                # 提供CSV和Excel两种下载格式
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 下载CSV格式",
                    data=csv,
                    file_name=f"{dataset_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # 生成Excel文件
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name=dataset_name)
                st.download_button(
                    label="📥 下载Excel格式",
                    data=output.getvalue(),
                    file_name=f"{dataset_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                # 自定义数据集下载（提供原始数据和处理后数据两种选项）
                download_option = st.radio(
                    "选择下载数据类型",
                    ["原始数据", "处理后数据"],
                    horizontal=True
                )
                
                if download_option == "原始数据" and st.session_state.raw_data is not None:
                    df = st.session_state.raw_data
                    filename_base = "自定义数据集_原始数据"
                    
                    # 提供CSV和Excel两种下载格式
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载CSV格式",
                        data=csv,
                        file_name=f"{filename_base}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name="数据集")
                    st.download_button(
                        label="📥 下载Excel格式",
                        data=output.getvalue(),
                        file_name=f"{filename_base}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                elif download_option == "处理后数据" and st.session_state.processed_data is not None:
                    df = st.session_state.processed_data
                    filename_base = "自定义数据集_处理后数据"
                    
                    # 提供CSV和Excel两种下载格式
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载CSV格式",
                        data=csv,
                        file_name=f"{filename_base}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name="数据集")
                    st.download_button(
                        label="📥 下载Excel格式",
                        data=output.getvalue(),
                        file_name=f"{filename_base}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    # 处理后数据尚未准备好时显示提示
                    if download_option == "处理后数据":
                        st.info("请先完成数据清洗并点击'准备训练数据'按钮以生成处理后数据")
        
        st.markdown("---")
        
        # 数据集详细信息
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.dataset_type == "built_in":
                # 使用name_key识别数据集
                name_key = st.session_state.dataset.get("name_key")
                dataset_name_map = {
                    "iris": "鸢尾花数据集",
                    "breast_cancer": "乳腺癌数据集",
                    "wine": "葡萄酒数据集",
                    "diabetes": "糖尿病数据集",
                    "california_housing": "加州房价数据集"
                }
                dataset_name = dataset_name_map.get(name_key, "未知数据集")
                
                st.subheader(f"{dataset_name} 基本信息")
                st.write(f"特征数量: {st.session_state.dataset.data.shape[1]}")
                st.write(f"样本数量: {st.session_state.dataset.data.shape[0]}")
                if st.session_state.task_type == "classification":
                    st.write(f"类别数量: {len(st.session_state.dataset.target_names)}")
                    st.write("类别名称: " + ", ".join(st.session_state.dataset.target_names))
                
                # 显示数据集前几行
                if name_key == "california_housing":
                    df = pd.DataFrame(
                        data=st.session_state.dataset.data,
                        columns=st.session_state.dataset.feature_names
                    )
                    df['目标值'] = st.session_state.dataset.target
                else:
                    df = pd.DataFrame(
                        data=st.session_state.dataset.data,
                        columns=st.session_state.dataset.feature_names
                    )
                    if st.session_state.task_type == "classification":
                        df['类别'] = [st.session_state.dataset.target_names[i] for i in st.session_state.dataset.target]
                    else:
                        df['目标值'] = st.session_state.dataset.target
                st.dataframe(df.head(10), use_container_width=True)
            else:
                st.subheader(f"自定义数据集 基本信息")
                # 显示原始数据信息
                st.write(f"原始数据 - 样本数量: {st.session_state.raw_data.shape[0]}")
                st.write(f"原始数据 - 列数量: {st.session_state.raw_data.shape[1]}")
                
                # 显示特征类型信息
                if st.session_state.numerical_cols or st.session_state.categorical_cols:
                    st.write(f"数值型特征数量: {len(st.session_state.numerical_cols)}")
                    st.write(f"类别型特征数量: {len(st.session_state.categorical_cols)}")
                
                # 只有当processed_data存在时才显示处理后的数据信息
                if st.session_state.processed_data is not None:
                    st.write(f"处理后数据 - 特征数量: {st.session_state.processed_data.shape[1] - 1}")  # 减1是因为包含目标列
                    st.write(f"处理后数据 - 样本数量: {st.session_state.processed_data.shape[0]}")
                    # 显示实际类别数（仅分类任务）
                    if st.session_state.task_type == "classification" and st.session_state.y_train is not None:
                        st.write(f"处理后数据 - 实际类别数: {len(np.unique(st.session_state.y_train))}")
                    st.write(f"目标列: {st.session_state.target_column if st.session_state.target_column is not None else '未选择'}")
                else:
                    st.info("请完成数据清洗并点击'准备训练数据'按钮以查看处理后的数据信息")
                
                # 显示数据集前几行（优先显示处理后的数据）
                if st.session_state.processed_data is not None:
                    st.write("处理后的数据预览:")
                    st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
                else:
                    st.write("原始数据预览:")
                    st.dataframe(st.session_state.raw_data.head(10), use_container_width=True)
        
        with col2:
            st.subheader("特征相关性分析")
            # 计算相关性矩阵
            if st.session_state.dataset_type == "built_in":
                corr_matrix = pd.DataFrame(st.session_state.dataset.data).corr()
                
                fig = px.imshow(
                    corr_matrix, 
                    labels=dict(color="相关性"),
                    text_auto=True,
                    color_continuous_scale="rdbu",
                    zmin=-1, zmax=1
                )
                fig.update_layout(
                    title='特征相关性热力图',
                    width=700,
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 检查是否有处理后的数据，否则使用原始数据
                if st.session_state.processed_data is not None:
                    # 只计算特征列之间的相关性，排除目标列
                    feature_cols = [col for col in st.session_state.processed_data.columns if col != st.session_state.target_column]
                    corr_matrix = st.session_state.processed_data[feature_cols].corr()
                    
                    # 绘制交互式热力图
                    fig = px.imshow(
                        corr_matrix, 
                        labels=dict(color="相关性"),
                        text_auto=True,
                        color_continuous_scale="rdbu",
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        title='特征相关性热力图 (处理后数据)',
                        width=700,
                        height=600,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # 使用原始数据计算相关性
                    numeric_cols = st.session_state.raw_data.select_dtypes(include=['number']).columns
                    if len(numeric_cols) >= 2:  # 需要至少两列数值型数据才能计算相关性
                        corr_matrix = st.session_state.raw_data[numeric_cols].corr()
                        
                        # 绘制交互式热力图
                        fig = px.imshow(
                            corr_matrix, 
                            labels=dict(color="相关性"),
                            text_auto=True,
                            color_continuous_scale="rdbu",
                            zmin=-1, zmax=1
                        )
                        fig.update_layout(
                            title='特征相关性热力图 (原始数据)',
                            width=700,
                            height=600,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("原始数据中数值型特征不足，无法计算相关性。请先完成数据处理步骤。")
        
        st.markdown("---")
        
        # 特征分布可视化
        st.header("📈 特征分布可视化")
        if st.session_state.feature_names is not None:
            feature_idx = st.selectbox(
                "选择特征查看分布",
                range(len(st.session_state.feature_names)),
                format_func=lambda x: st.session_state.feature_names[x]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                # 特征分布直方图
                if st.session_state.dataset_type == "built_in":
                    data = st.session_state.dataset.data[:, feature_idx]
                else:
                    data = st.session_state.processed_data[st.session_state.feature_names[feature_idx]].values
                
                fig = px.histogram(
                    x=data,
                    nbins=30,
                    labels={'x': st.session_state.feature_names[feature_idx]},
                    title=f'{st.session_state.feature_names[feature_idx]} 分布',
                    marginal='box'  # 添加箱线图作为边际分布
                )
                fig.update_layout(
                    width=600,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 特征按目标值分布
                if st.session_state.dataset_type == "built_in":
                    if st.session_state.task_type == "classification":
                        fig = px.box(
                            x=st.session_state.dataset.target,
                            y=st.session_state.dataset.data[:, feature_idx],
                            labels={'x': '类别', 'y': st.session_state.feature_names[feature_idx]},
                            title=f'{st.session_state.feature_names[feature_idx]} 按类别分布'
                        )
                        fig.update_layout(
                            width=600,
                            height=400
                        )
                        # 更新x轴标签
                        if st.session_state.target_names is not None:
                            fig.update_xaxes(
                                ticktext=st.session_state.target_names,
                                tickvals=list(range(len(st.session_state.target_names)))
                            )
                    else:  # regression
                        fig = px.scatter(
                            x=st.session_state.dataset.target,
                            y=st.session_state.dataset.data[:, feature_idx],
                            labels={'x': '目标值', 'y': st.session_state.feature_names[feature_idx]},
                            title=f'{st.session_state.feature_names[feature_idx]} 与目标值关系',
                            trendline='ols'  # 添加趋势线
                        )
                        fig.update_layout(
                            width=600,
                            height=400
                        )
                else:
                    if st.session_state.task_type == "classification":
                        fig = px.box(
                            x=st.session_state.processed_data[st.session_state.target_column],
                            y=st.session_state.processed_data[st.session_state.feature_names[feature_idx]],
                            labels={'x': '类别', 'y': st.session_state.feature_names[feature_idx]},
                            title=f'{st.session_state.feature_names[feature_idx]} 按类别分布'
                        )
                        fig.update_layout(
                            width=600,
                            height=400
                        )
                        # 更新x轴标签
                        if st.session_state.target_names is not None:
                            fig.update_xaxes(
                                ticktext=st.session_state.target_names,
                                tickvals=list(range(len(st.session_state.target_names)))
                            )
                    else:  # regression
                        fig = px.scatter(
                            x=st.session_state.processed_data[st.session_state.target_column],
                            y=st.session_state.processed_data[st.session_state.feature_names[feature_idx]],
                            labels={'x': '目标值', 'y': st.session_state.feature_names[feature_idx]},
                            title=f'{st.session_state.feature_names[feature_idx]} 与目标值关系',
                            trendline='ols'  # 添加趋势线
                        )
                        fig.update_layout(
                            width=600,
                            height=400
                        )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # 当特征名称尚未加载时显示提示
            if st.session_state.dataset_type == "custom" and st.session_state.raw_data is not None:
                st.info("请完成数据清洗并点击'准备训练数据'按钮以查看特征分布可视化")
        
        st.markdown("---")
        
        # 模型训练结果
        if st.session_state.is_trained:
            st.header("🏆 模型训练结果")
            
            # 模型性能概览
            st.subheader(f"{st.session_state.model_name} 性能指标")
            
            if st.session_state.task_type == "classification":
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("训练准确率", f"{st.session_state.train_metrics.get('accuracy', 0):.4f}")
                with col2:
                    st.metric("训练精确率", f"{st.session_state.train_metrics.get('precision', 0):.4f}")
                with col3:
                    st.metric("训练召回率", f"{st.session_state.train_metrics.get('recall', 0):.4f}")
                with col4:
                    st.metric("训练F1分数", f"{st.session_state.train_metrics.get('f1', 0):.4f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("测试准确率", f"{st.session_state.test_metrics.get('accuracy', 0):.4f}")
                with col2:
                    st.metric("测试精确率", f"{st.session_state.test_metrics.get('precision', 0):.4f}")
                with col3:
                    st.metric("测试召回率", f"{st.session_state.test_metrics.get('recall', 0):.4f}")
                with col4:
                    st.metric("测试F1分数", f"{st.session_state.test_metrics.get('f1', 0):.4f}")
            else:  # regression
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("训练MSE", f"{st.session_state.train_metrics.get('mse', 0):.4f}")
                with col2:
                    st.metric("训练RMSE", f"{st.session_state.train_metrics.get('rmse', 0):.4f}")
                with col3:
                    st.metric("训练MAE", f"{st.session_state.train_metrics.get('mae', 0):.4f}")
                with col4:
                    st.metric("训练R²分数", f"{st.session_state.train_metrics.get('r2', 0):.4f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("测试MSE", f"{st.session_state.test_metrics.get('mse', 0):.4f}")
                with col2:
                    st.metric("测试RMSE", f"{st.session_state.test_metrics.get('rmse', 0):.4f}")
                with col3:
                    st.metric("测试MAE", f"{st.session_state.test_metrics.get('mae', 0):.4f}")
                with col4:
                    st.metric("测试R²分数", f"{st.session_state.test_metrics.get('r2', 0):.4f}")
            
            st.write(f"**训练时间:** {st.session_state.training_time:.4f} 秒")
            
            st.markdown("---")
            
            # 详细评估指标
            st.subheader("详细评估报告")
            if st.session_state.task_type == "classification":
                # 获取实际存在的类别
                actual_classes = np.unique(st.session_state.y_test)
                report = classification_report(
                    st.session_state.y_test,
                    st.session_state.y_pred,
                    labels=actual_classes,  # 明确指定实际存在的类别
                    target_names=st.session_state.target_names,
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            else:
                # 回归任务的评估报告
                metrics_df = pd.DataFrame({
                    '指标': ['MSE', 'RMSE', 'MAE', 'R²分数'],
                    '训练集': [
                        f"{st.session_state.train_metrics.get('mse', 0):.4f}",
                        f"{st.session_state.train_metrics.get('rmse', 0):.4f}",
                        f"{st.session_state.train_metrics.get('mae', 0):.4f}",
                        f"{st.session_state.train_metrics.get('r2', 0):.4f}"
                    ],
                    '测试集': [
                        f"{st.session_state.test_metrics.get('mse', 0):.4f}",
                        f"{st.session_state.test_metrics.get('rmse', 0):.4f}",
                        f"{st.session_state.test_metrics.get('mae', 0):.4f}",
                        f"{st.session_state.test_metrics.get('r2', 0):.4f}"
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True)
            
            st.markdown("---")
            
            # 可视化结果
            st.subheader("结果可视化")
            
            # 混淆矩阵（仅分类任务）
            if st.session_state.task_type == "classification":
                st.subheader("混淆矩阵")
                cm_fig = plot_confusion_matrix(
                    st.session_state.y_test,
                    st.session_state.y_pred,
                    st.session_state.target_names
                )
                st.plotly_chart(cm_fig, use_container_width=True)
            
            # 学习曲线
            st.subheader("学习曲线")
            lc_fig = plot_learning_curve(
                st.session_state.model,
                st.session_state.X_train,
                st.session_state.y_train,
                f"{st.session_state.model_name} 学习曲线"
            )
            st.plotly_chart(lc_fig, use_container_width=True)
            
            # 特征重要性（如果模型支持）
            if hasattr(st.session_state.model, 'feature_importances_'):
                st.subheader("特征重要性")
                fi_fig = plot_feature_importance(
                    st.session_state.model,
                    st.session_state.feature_names
                )
                if fi_fig:
                    st.plotly_chart(fi_fig, use_container_width=True)
            
            # ROC曲线（仅二分类问题）
            if st.session_state.task_type == "classification" and len(np.unique(st.session_state.y_test)) == 2:
                st.subheader("ROC曲线")
                roc_fig = plot_roc_curve(
                    st.session_state.model,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    st.session_state.target_names
                )
                if roc_fig:
                    st.plotly_chart(roc_fig, use_container_width=True)
            
            # 回归任务的可视化
            if st.session_state.task_type == "regression":
                st.subheader("预测值 vs 真实值")
                pred_fig = plot_prediction_vs_actual(
                    st.session_state.y_test,
                    st.session_state.y_pred
                )
                st.plotly_chart(pred_fig, use_container_width=True)
                
                st.subheader("残差图")
                residual_fig = plot_residuals(
                    st.session_state.y_test,
                    st.session_state.y_pred
                )
                st.plotly_chart(residual_fig, use_container_width=True)
            
            st.markdown("---")
            
            # 预测示例
            st.subheader("预测示例")
            if len(st.session_state.X_test) > 0:
                sample_idx = st.slider(
                    "选择测试集中的样本查看预测结果",
                    0, len(st.session_state.X_test) - 1, 0
                )
                
                sample = st.session_state.X_test[sample_idx:sample_idx+1]
                true_label = st.session_state.y_test[sample_idx]
                pred_label = st.session_state.y_pred[sample_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**样本特征值:**")
                    features_df = pd.DataFrame(
                        sample, 
                        columns=st.session_state.feature_names
                    )
                    st.dataframe(features_df, use_container_width=True)
                
                with col2:
                    st.write("**预测结果:**")
                    if st.session_state.task_type == "classification":
                        actual_classes = np.unique(st.session_state.y_test)
                        if true_label == pred_label:
                            st.success(f"正确预测: {st.session_state.target_names[np.where(actual_classes == true_label)[0][0]]}")
                        else:
                            st.error(f"预测错误: 预测为 {st.session_state.target_names[np.where(actual_classes == pred_label)[0][0]]}, 实际为 {st.session_state.target_names[np.where(actual_classes == true_label)[0][0]]}")
                        
                        # 显示预测概率（如果模型支持）
                        if hasattr(st.session_state.model, 'predict_proba'):
                            proba = st.session_state.model.predict_proba(sample)[0]
                            proba_df = pd.DataFrame({
                                '类别': st.session_state.target_names,
                                '概率': [f"{p:.4f}" for p in proba]
                            })
                            st.dataframe(proba_df, use_container_width=True)
                    else:  # regression
                        st.write(f"真实值: {true_label:.4f}")
                        st.write(f"预测值: {pred_label:.4f}")
                        st.write(f"误差: {abs(true_label - pred_label):.4f}")
        
        else:
            # 提示训练模型
            st.info("""
            ### 等待模型训练
            
            请在侧边栏选择一个机器学习模型并点击"开始训练"按钮。
            
            训练完成后，这里将显示模型性能指标和可视化结果。
            """)
            st.image("https://images.unsplash.com/photo-1517694712202-14dd9538aa97?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", 
                     caption="机器学习模型训练")
        
        # 训练历史记录
        if st.session_state.training_history:
            st.markdown("---")
            st.header("📜 训练历史记录")
            with st.expander(f"查看 {len(st.session_state.training_history)} 条训练记录"):
                for i, entry in enumerate(reversed(st.session_state.training_history)):
                    st.subheader(f"训练记录 {len(st.session_state.training_history) - i} ({entry['timestamp']})")
                    st.write(f"模型: {entry['model_name']}")
                    st.write(f"训练时间: {entry['training_time']:.4f}秒")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**训练集指标:**")
                        metrics_df = pd.DataFrame(list(entry['train_metrics'].items()), columns=['指标', '值'])
                        metrics_df['值'] = metrics_df['值'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(metrics_df, use_container_width=True)
                    
                    with col2:
                        st.write("**测试集指标:**")
                        metrics_df = pd.DataFrame(list(entry['test_metrics'].items()), columns=['指标', '值'])
                        metrics_df['值'] = metrics_df['值'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(metrics_df, use_container_width=True)
                    
                    st.markdown("---")
    
    # 页脚信息
    st.markdown("---")
    st.caption("2025.8.20版本 | 升级了内置数据集机器学习及其可视化功能, 自定义数据集功能仍在调试中")

if __name__ == "__main__":
    main()
