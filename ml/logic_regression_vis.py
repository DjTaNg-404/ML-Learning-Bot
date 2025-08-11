import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ...existing code...
def logic_regression_vis(X_train, X_test, y_train, y_test, feature_names, target_names, max_iter, C, random_state):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=max_iter, C=C, random_state=random_state, multi_class="ovr")
    model.fit(X_train_scaled, y_train)

    st.write("### 变量重要性（特征系数）")
    coef = model.coef_
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(coef.shape[0]):
        ax.bar(feature_names, coef[i], alpha=0.6, label=f"类别 {target_names[i]}")
    ax.set_ylabel("系数")
    ax.set_title("各类别特征系数")
    ax.legend()
    st.pyplot(fig)

    st.write("### 测试集准确率")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**{acc:.2f}**")

    st.write("### 混淆矩阵")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax2)
    ax2.set_xlabel("预测标签")
    ax2.set_ylabel("真实标签")
    st.pyplot(fig2)

    st.write("### ROC 曲线（每个类别）")
    fig3, ax3 = plt.subplots()
    for i in range(len(target_names)):
        y_test_bin = (y_test == i).astype(int)
        y_score = model.predict_proba(X_test_scaled)[:, i]
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f"{target_names[i]} (AUC={roc_auc:.2f})")
    ax3.plot([0, 1], [0, 1], "k--", lw=1)
    ax3.set_xlabel("假阳性率")
    ax3.set_ylabel("真阳性率")
    ax3.set_title("ROC 曲线")
    ax3.legend()
    st.pyplot(fig3)
