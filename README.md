# Credit Score Classification (信用评分分类项目)

## Project Overview / 项目概述

This project focuses on credit score classification tasks, aiming to build an accurate and robust model to predict customers' credit scores (Excellent/Good/Poor) based on comprehensive features, including personal information, financial behavior, and credit history. Addressing core challenges such as data imbalance, feature redundancy, and model overfitting, the project integrates systematic data preprocessing, multi-dimensional feature engineering, and extensive model comparison to provide a reliable solution for financial credit risk assessment.

The model is trained and validated on a real-world dataset with 15,000 customer records and 28 features. Through multi-round optimization (class balancing, feature selection, hyperparameter tuning, and ensemble fusion), the final integrated model achieves 91.2% classification accuracy and 90.5% macro F1-score, significantly outperforming single baseline models. It effectively supports financial institutions in the efficient, transparent evaluation of creditworthiness.

本项目聚焦信用评分分类任务，旨在基于个人信息、金融行为、信用历史等综合特征，构建精准稳健的信用评分预测模型（优秀/良好/较差三级分类）。针对数据不平衡、特征冗余、模型过拟合等核心问题，整合系统数据预处理、多维度特征工程与全面模型对比，为金融信用风险评估提供可靠解决方案。

模型基于含15,000条客户记录、28项特征的真实数据集完成训练与验证，经多轮优化（类别平衡、特征筛选、超参数调优、集成融合）后，最终集成模型达成91.2%的分类准确率与90.5%的宏平均F1分数，显著优于单一基准模型，有效支撑金融机构高效、透明地开展信用资质评估。

## Key Models & Performance / 核心模型与性能

| Model Category          | Model Name                | Test Accuracy | Macro F1-Score | Key Optimization & Notes                                                                 |
|-------------------------|---------------------------|---------------|----------------|-----------------------------------------------------------------------------------------|
| Baseline Models         | Logistic Regression       | 76.2%         | 74.8%          | Standardization, L1 regularization; handles linear relationships well                   |
|                         | K-Nearest Neighbors (KNN) | 78.5%         | 76.3%          | K=7, distance-weighted; sensitive to feature scaling                                   |
|                         | Decision Tree             | 80.1%         | 78.6%          | Max_depth=8, min_samples_split=20; simple interpretability                             |
| Ensemble Models         | Basic Random Forest       | 82.5%         | 80.1%          | 100 estimators, default hyperparameters                                                 |
|                         | Tuned Random Forest       | 87.3%         | 85.9%          | Max_depth=12, max_features='sqrt', n_estimators=250; reduced overfitting               |
| Gradient Boosting Models| Tuned XGBoost             | 89.7%         | 88.3%          | Learning rate=0.08, max_depth=6, n_estimators=300, subsample=0.9                       |
|                         | Tuned LightGBM            | 88.9%         | 87.5%          | Num_leaves=31, subsample=0.8, colsample_bytree=0.8, early stopping                     |
|                         | Tuned CatBoost            | 87.8%         | 86.9%          | Class weight adjustment, learning rate=0.1, max_depth=5                                 |
| Deep Learning           | Multi-Layer Perceptron (MLP) | 85.2%      | 83.7%          | [512,256,128] hidden layers, dropout=0.3, batch size=64                                |
| Ensemble Fusion         | Stacking (XGBoost+LightGBM+Tuned RF) | 91.2% | 90.5% | Meta-model: Logistic Regression; fuses strengths of multiple models                     |


| 模型类别                | 模型名称                  | 测试准确率  | 宏平均F1分数 | 核心优化与说明                                                                          |
|-------------------------|---------------------------|-------------|--------------|-----------------------------------------------------------------------------------------|
| 基准模型                | 逻辑回归                  | 76.2%       | 74.8%        | 特征标准化、L1正则化；擅长捕捉线性关系                                                  |
|                         | K近邻（KNN）              | 78.5%       | 76.3%        | K=7、距离加权；对特征缩放敏感                                                            |
|                         | 决策树                    | 80.1%       | 78.6%        | 最大深度=8、最小样本分裂数=20；可解释性强                                                |
| 集成模型                | 基础随机森林              | 82.5%       | 80.1%        | 100个基学习器、默认超参数                                                              |
|                         | 调优后随机森林            | 87.3%       | 85.9%        | 最大深度=12、最大特征数='sqrt'、基学习器数=250；降低过拟合                              |
| 梯度提升模型            | 调优后XGBoost             | 89.7%       | 88.3%        | 学习率=0.08、最大深度=6、基学习器数=300、子样本比例=0.9                                |
|                         | 调优后LightGBM            | 88.9%       | 87.5%        | 叶节点数=31、子样本比例=0.8、列采样比例=0.8、早停机制                                  |
|                         | 调优后CatBoost            | 87.8%       | 86.9%        | 类别权重调整、学习率=0.1、最大深度=5                                                    |
| 深度学习模型            | 多层感知机（MLP）         | 85.2%       | 83.7%        | [512,256,128]隐藏层、dropout=0.3、批大小=64                                             |
| 集成融合模型            | 堆叠集成（XGBoost+LightGBM+调优后随机森林） | 91.2% | 90.5% | 元模型：逻辑回归；融合多模型优势                                                        |

## Core Features / 核心特点

1. Comprehensive Data Preprocessing: Handles missing values (median/mode imputation), outliers (IQR method + capping), and class imbalance (SMOTE oversampling + class weight adjustment) to ensure data quality. 全面数据预处理：采用中位数/众数插补缺失值、IQR法+数值封顶处理异常值、SMOTE过采样+类别权重调整解决数据不平衡，保障数据质量。

2. Multi-Dimensional Feature Engineering: Implements feature selection (mutual information + correlation analysis), dimensionality reduction (PCA), and feature encoding (one-hot for categorical features, Z-score for numerical features), reducing redundant features by 35%. 多维度特征工程：通过互信息法+相关性分析筛选特征、PCA降维、分类特征独热编码/数值特征Z-score标准化，减少35%冗余特征。

3. Extensive Model Comparison: Covers 10+ models across baseline, ensemble, gradient boosting, and deep learning categories, providing a full-view performance reference. 全面模型对比：覆盖基准、集成、梯度提升、深度学习四大类10+模型，提供全景式性能参考。

4. Systematic Optimization Pipeline: Standardizes hyperparameter tuning (GridSearchCV + RandomizedSearchCV) and model validation (5-fold cross-validation) to ensure result reliability. 系统化优化流程：标准化超参数调优（网格搜索+随机搜索）与模型验证（5折交叉验证），确保结果可靠性。

5. High Interpretability & Practicality: Outputs feature importance ranking (SHAP analysis) and confusion matrix visualization, supporting transparent credit decision-making; the final model is lightweight and suitable for production deployment. 高可解释性与实用性：输出特征重要性排序（SHAP分析）与混淆矩阵可视化，支撑透明化信用决策；最终模型轻量化，适配生产部署。

## Data Source / 数据来源

The dataset is derived from a public credit evaluation database (compliant with GDPR/CCPA privacy regulations), containing 15,000 customer records and 28 features across three categories:
• Personal Information: Age, gender, occupation, education level, marital status, household size.
• Financial Behavior: Credit card usage frequency, average monthly spending, loan repayment history (on-time/late/default), savings balance, debt-to-income ratio.
• Credit Labels: 3-level classification (Excellent/Good/Poor) based on comprehensive credit evaluation.
All data is de-identified to protect customer privacy, with no sensitive personal information retained.

数据集来源于公开信用评估数据库（符合GDPR/CCPA隐私规范），包含15,000条客户记录与28项特征，涵盖三大类别：
• 个人信息：年龄、性别、职业、教育水平、婚姻状况、家庭人口数。
• 金融行为：信用卡使用频率、月均消费额、贷款还款记录（按时/逾期/违约）、储蓄余额、收入负债比。
• 信用标签：基于综合信用评估的三级分类（优秀/良好/较差）。
数据已完成去标识化处理，保护客户隐私，无敏感个人信息留存。

## Project Structure / 项目结构
```
Credit-Score-Classification/
├── 5003_final_report_W04G06-1.html   # Project final report (includes full research details, results & analysis)
├── ASS2_final.Rmd                    # Core code file: data preprocessing, feature engineering, model training & validation
├── README.md                         # Project documentation (English/Chinese)
```

