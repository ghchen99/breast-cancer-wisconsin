# %% [markdown]
# # Cancer Diagnosis Prediction

# %% [markdown]
# This notebook focuses on predicting whether a breast cancer tumor is malignant or benign based on features computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. The dataset describes characteristics of cell nuclei present in the image, such as:
# 
# - **Radius**: Mean of distances from center to points on the perimeter.
# - **Texture**: Standard deviation of gray-scale values.
# - **Perimeter** and **Area**.
# - **Smoothness**: Local variation in radius lengths.
# - **Compactness**: Calculated as: (perimeter² / area - 1.0).
# - **Concavity**: Severity of concave portions of the contour.
# - **Concave Points**: Number of concave portions of the contour.
# - **Symmetry** and **Fractal Dimension**: "Coastline approximation" - 1.
# 
# By analysing these features, the goal is to develop a machine learning pipeline to predict tumor diagnosis accurately.

# %% [markdown]
# ## Load Dataset

# %% [markdown]
# ### Import Necessary Libraries

# %%
import pandas as pd
import numpy as np
import joblib
import os
import kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                           roc_curve, auc)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# Preprocessing Steps:
# 1. Download dataset using kagglehub API
# 2. Convert ID column to string type for consistency
# 3. Remove empty/unnamed columns
# 4. Save cleaned dataset to local directory for model use

# %%
# Goes up one level from notebooks folder
os.makedirs('../data', exist_ok=True)

# Read and process as before
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
df = pd.read_csv(path + '/data.csv')
df['id'] = df['id'].astype(str)
df.set_index('id', inplace=True)
df = df.drop('Unnamed: 32', axis=1)

# Save to parent directory
output_path = '../data/cell-data.csv'
df.to_csv(output_path, index=False)
df.head()

# %%
df.info()

# %% [markdown]
# Dataset Information:
# - Source: UCI Machine Learning Repository
# - Features: 30 numeric features computed from cell nuclei images
# - Target: Binary classification (M = malignant, B = benign)
# - Size: 569 instances
# 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
# They describe characteristics of the cell nuclei present in the image.
# 
# Feature Categories:
# 1. Radius (mean of distances from center to points on the perimeter)
# 2. Texture (standard deviation of gray-scale values)
# 3. Perimeter
# 4. Area
# 5. Smoothness (local variation in radius lengths)
# 6. Compactness (perimeter^2 / area - 1.0)
# 7. Concavity (severity of concave portions of the contour)
# 8. Concave points (number of concave portions of the contour)
# 9. Symmetry
# 10. Fractal dimension ("coastline approximation" - 1)
# 
# For each feature, three measurements are provided:
# - Mean
# - Standard error (SE)
# - "Worst" or largest (mean of the three largest values)

# %%
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# %%
# Basic statistics of numerical columns
print("\nNumerical Features Statistics:")
print(df.describe())

# %%
# Distribution of target variable
print("\nTarget Variable Distribution:")
print(df['diagnosis'].value_counts(normalize=True))

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Correlation

# %% [markdown]
# We visualise the correlation matrix between features to:
# 
# 1. Identify Multicollinearity:
#    - Detect highly correlated features that might provide redundant information
#    - Help prevent model overfitting by identifying features that could potentially be removed
#    - Particularly important for this dataset as many features are derived from the same underlying measurements
# 
# 2. Understand Feature Relationships:
#    - Reveal patterns between different cell measurements (radius, texture, perimeter, etc.)
#    - Show relationships between mean, SE, and "worst" measurements of the same characteristic
#    - Help identify which features might be most informative for diagnosis
# 
# 3. Guide Feature Selection:
#    - Inform decisions about which features to keep or combine
#    - Help identify groups of related features that might be candidates for dimensionality reduction
#    - Support the creation of more interpretable and efficient models
# 

# %%
# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

# Create correlation matrix heatmap
plt.figure(figsize=(15, 12))
correlation_matrix = df.drop(['diagnosis'], axis=1).corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()

# Save plot before showing it
plt.savefig('../results/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Identify highly correlated feature pairs
def get_highly_correlated_pairs(correlation_matrix, threshold=0.8):
    """
    Find pairs of features with correlation above threshold
    """
    highly_correlated = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                highly_correlated.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': correlation_matrix.iloc[i, j]
                })
    return pd.DataFrame(highly_correlated).sort_values('correlation', ascending=False)

# Perform the analysis
correlation_matrix = df.drop(['diagnosis'], axis=1).corr()

# Get highly correlated feature pairs
high_correlations = get_highly_correlated_pairs(correlation_matrix)
print("\nHighly Correlated Feature Pairs (correlation > 0.8):")
display(high_correlations)

# %%
# Calculate correlation with target
def get_target_correlations(df):
    """
    Calculate correlation of features with the diagnosis
    """
    # Convert diagnosis to numeric (0 for 'B', 1 for 'M')
    diagnosis_numeric = (df['diagnosis'] == 'M').astype(int)

    # Calculate correlation with each feature
    correlations = []
    for column in df.drop(['diagnosis'], axis=1).columns:
        correlation = df[column].corr(diagnosis_numeric)
        correlations.append({
            'feature': column,
            'correlation_with_diagnosis': abs(correlation)
        })
    return pd.DataFrame(correlations).sort_values('correlation_with_diagnosis', ascending=False)

# Get feature correlations with diagnosis
target_correlations = get_target_correlations(df)
print("\nFeature Correlations with Diagnosis (absolute values):")
display(target_correlations)

# %%
# Group features by measurement type
def group_related_features(df):
    """
    Group features by their base measurement (radius, texture, etc.)
    """
    feature_groups = {}
    base_measurements = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                        'compactness', 'concavity', 'concave points', 'symmetry',
                        'fractal_dimension']

    for base in base_measurements:
        related_features = [col for col in df.columns if base in col]
        feature_groups[base] = related_features

    return feature_groups

# Group related features
feature_groups = group_related_features(df)
print("\nFeature Groups:")
for base, features in feature_groups.items():
    print(f"\n{base.capitalize()} measurements:")
    print(features)

# %%
# Calculate mean correlation within groups
print("\nMean Absolute Correlation within Feature Groups:")
for base, features in feature_groups.items():
    # Example: base might be 'radius',
    # features would be ['radius_mean', 'radius_se', 'radius_worst']

    # correlation_matrix.loc[features, features] creates a sub-matrix of
    # correlations just for these related features

    # For radius features, it creates a 3x3 matrix like:
    #              radius_mean  radius_se  radius_worst
    # radius_mean      1.0        0.3         0.7
    # radius_se        0.3        1.0         0.4
    # radius_worst     0.7        0.4         1.0

    # .abs() takes absolute values of correlations
    # first .mean() averages each row
    # second .mean() averages those averages
    group_corr = correlation_matrix.loc[features, features].abs().mean().mean()
    print(f"{base}: {group_corr:.3f}")

# %% [markdown]
# **Feature Selection Strategy**
# 
# If two features are highly correlated (>0.9), we can keep the feature that has stronger predictive power for diagnosis. For example, if 'radius_mean' and 'perimeter_mean' are highly correlated, we keep whichever has higher correlation with malignant/benign outcome
# 
# The final output provides:
# - List of most important features for diagnosis.
# - List of features that could be removed without significant loss of information
# This supports building a more efficient model with reduced multicollinearity.

# %%
# Feature selection recommendations
high_importance_features = target_correlations[target_correlations['correlation_with_diagnosis'] > 0.5]['feature'].tolist()

# Identify features to potentially remove due to high correlation
features_to_consider_removing = []
correlation_threshold = 0.9
for _, row in high_correlations[high_correlations['correlation'] > correlation_threshold].iterrows():
    # Keep the feature with higher correlation to diagnosis
    feature1_corr = target_correlations[target_correlations['feature'] == row['feature1']]['correlation_with_diagnosis'].iloc[0]
    feature2_corr = target_correlations[target_correlations['feature'] == row['feature2']]['correlation_with_diagnosis'].iloc[0]

    if feature1_corr > feature2_corr:
        features_to_consider_removing.append(row['feature2'])
    else:
        features_to_consider_removing.append(row['feature1'])

features_to_consider_removing = list(set(features_to_consider_removing))

print("\nFeature Selection Recommendations:")
print("\nHighly Important Features (correlation with diagnosis > 0.5):")
print(high_importance_features)
print("\nFeatures to Consider Removing (due to high correlation):")
print(features_to_consider_removing)

# %% [markdown]
# **[insert comparison of model trained on original dataset vs reduced dataset]**

# %% [markdown]
# ### Box Plot Analysis and Distribution Interpretation
# 
# 1. Basic Distribution Statistics:
#    - Q1 (25th percentile): Lower edge of box
#    - Q3 (75th percentile): Upper edge of box
#    - IQR (Interquartile Range) = Q3 - Q1: Height of box
#    - Median: Line inside box
#    - Whiskers: Extend to last point within 1.5 * IQR of Q1 and Q3
#    
# 2. Outlier Detection:
#    - Lower bound = Q1 - 1.5 * IQR
#    - Upper bound = Q3 + 1.5 * IQR
#    - Points beyond these bounds are considered outliers

# %%
# Create separate box plots for mean, worst, and se features
plt.figure(figsize=(15, 25))

# Mean features
plt.subplot(3, 1, 1)
mean_features = [col for col in df.columns if 'mean' in col]
df.boxplot(column=mean_features)
plt.xticks(rotation=90)
plt.title('Distribution of Mean Features')

# SE features
plt.subplot(3, 1, 2)
se_features = [col for col in df.columns if '_se' in col]
df.boxplot(column=se_features)
plt.xticks(rotation=90)
plt.title('Distribution of SE Features')

# Worst features
plt.subplot(3, 1, 3)
worst_features = [col for col in df.columns if 'worst' in col]
df.boxplot(column=worst_features)
plt.xticks(rotation=90)
plt.title('Distribution of Worst Features')

plt.tight_layout()

# Save plot before showing it
plt.savefig('../results/boxplot_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# 1. Outlier Detection:
#    - Code calculates percentage of points that are outliers.
# 
# 2. Variation Measures:
#    - Standard Deviation (std): Measures spread of all data points
#    - IQR: Robust measure of spread, less sensitive to outliers
#    - High std + high IQR: Feature has wide spread
#    - Low std + low IQR: Feature might be redundant
# 
# 3. Feature Selection Criteria:
#    - High Outlier Percentage (>10%):
#      May indicate noisy or problematic features
#    - Low Variation (< 0.1 * median std):
#      May indicate feature doesn't provide much discriminative power
#    - SE vs Mean/Worst Comparison:
#      If SE variation < 30% of mean/worst, SE feature might be redundant
# 
# 4. Measurement Type Comparison:
#    - Compares variation across mean, SE, and worst for each measurement
#    - Helps identify which measurement type (mean/SE/worst) is most informative
#    - SE measurements often show less variation, making them candidates for removal
# 
# 5. Final Recommendations:
#    - Suggests dropping features with very low variation
#    - Identifies redundant SE measurements
#    - Creates list of features that might not contribute significantly to model

# %%
# Analyze outliers and spread for each feature type
def analyze_feature_distributions(df, feature_group_name, features):
    """
    Analyze distribution characteristics of feature groups
    """
    stats = []
    for col in features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))][col].count()

        stats.append({
            'feature': col,
            'std': df[col].std(),
            'iqr': iqr,
            'outlier_count': outliers,
            'outlier_percentage': (outliers / len(df)) * 100
        })

    stats_df = pd.DataFrame(stats)
    print(f"\n{feature_group_name} Features Analysis:")
    display(stats_df.sort_values('outlier_percentage', ascending=False))
    return stats_df

# Analyze each feature group
mean_stats = analyze_feature_distributions(df, "Mean", mean_features)
se_stats = analyze_feature_distributions(df, "SE", se_features)
worst_stats = analyze_feature_distributions(df, "Worst", worst_features)

# Identify features with high outlier percentages or low variation
outlier_threshold = 10  # Features with > 10% outliers
variation_threshold = df[mean_features + se_features + worst_features].std().median() * 0.1  # Features with very low variation

features_to_consider_dropping = []

# Check for features with high outliers
high_outlier_features = pd.concat([mean_stats, se_stats, worst_stats])[
    pd.concat([mean_stats, se_stats, worst_stats])['outlier_percentage'] > outlier_threshold
]['feature'].tolist()

# Check for features with low variation
low_variation_features = df[mean_features + se_features + worst_features].columns[
    df[mean_features + se_features + worst_features].std() < variation_threshold
].tolist()

# Compare variations between mean, SE, and worst for same measurements
measurement_comparisons = {}
base_measurements = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                    'compactness', 'concavity', 'concave points', 'symmetry',
                    'fractal_dimension']

for base in base_measurements:
    mean_var = df[f'{base}_mean'].std() if f'{base}_mean' in df.columns else 0
    se_var = df[f'{base}_se'].std() if f'{base}_se' in df.columns else 0
    worst_var = df[f'{base}_worst'].std() if f'{base}_worst' in df.columns else 0

    measurement_comparisons[base] = {
        'mean_var': mean_var,
        'se_var': se_var,
        'worst_var': worst_var
    }

print("\nFeature Selection Recommendations based on Distributions:")
print("\nFeatures with high outlier percentage (>10%):")
print(high_outlier_features)

print("\nFeatures with low variation:")
print(low_variation_features)

print("\nVariation Comparison for Each Measurement Type:")
for base, vars in measurement_comparisons.items():
    print(f"\n{base}:")
    print(f"Mean variation: {vars['mean_var']:.3f}")
    print(f"SE variation: {vars['se_var']:.3f}")
    print(f"Worst variation: {vars['worst_var']:.3f}")

# Final recommendations
print("\nFinal Recommendations:")
print("Consider dropping these features based on distribution analysis:")
redundant_features = []

for base in base_measurements:
    vars = measurement_comparisons[base]
    # If SE variation is much lower than mean and worst, consider dropping SE
    if vars['se_var'] < 0.3 * vars['mean_var'] and vars['se_var'] < 0.3 * vars['worst_var']:
        redundant_features.append(f"{base}_se")

# Add features with very low variation
redundant_features.extend(low_variation_features)

# Remove duplicates and sort
redundant_features = sorted(list(set(redundant_features)))
print(redundant_features)

# %% [markdown]
# ## Model Training

# %%
# Separate features and target
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# %% [markdown]
# **Random Forest Working Principle**:
# 
# - Creates multiple decision trees (100 or 200 in this case)
# - Each tree:
#   - a. Gets a random subset of the data (bootstrap sampling)
#   - b. At each split, considers a random subset of features
#   - c. Makes its own prediction (0 or 1)
# - Final prediction is majority vote from all trees

# %%
# Convert target to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) # 'M' -> 1, 'B' -> 0

# Split the data, stratify=y ensures proportional split of malignant/benign cases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create preprocessing and modeling pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), # StandardScaler standardises features to mean=0, std=1
    ('classifier', RandomForestClassifier(random_state=42)) # Combines multiple decision trees
])

# Define hyperparameters for grid search
param_grid = {
    'classifier__n_estimators': [100, 200], # Number of trees in the forest
    'classifier__max_depth': [10, 20, None], # Maximum depth of each tree
    'classifier__min_samples_split': [2, 5], # Minimum samples required to split node
    'classifier__min_samples_leaf': [1, 2] # Minimum samples required in leaf node
}

# Perform grid search with cross-validation
# Tests all combinations of hyperparameters
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# %% [markdown]
# **Why cross-validation?**
# 
# 
# For a 650 x 30 dataset, cross-validation is the best choice because:
# 
# 1. Sample Size Per Split:
#   - With a validation set approach (60/20/20):
#     - 390 training samples, 130 validation samples, 130 test samples
#     - Example validation scores might be:
#     
#         Model Config A: 0.85 (but could be anywhere from 0.80-0.90 due to random split)
#         
#         Model Config B: 0.87 (but could be anywhere from 0.82-0.92 due to random split)
# 
#   - With 5-fold CV:
#     - 520 samples for training/validation
#     - Each fold uses ~416 samples for training
#     - Example scores for same configs:
#         
#         Model Config A: 0.83, 0.86, 0.84, 0.87, 0.85 (avg: 0.85 ± 0.015)
#         
#         Model Config B: 0.84, 0.85, 0.84, 0.85, 0.84 (avg: 0.844 ± 0.005)
# 
#   **Now we can be more confident Config A is actually better.**
# 
# 2. Feature to Sample Ratio:
#   - You have 30 features
#   - Rule of thumb: want at least 10 samples per feature for stable estimates
#   - Validation split gives only: 390/30 = 13 samples per feature
#   - CV gives: 416/30 = ~14 samples per feature, but with 5 different evaluations

# %%
# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Evaluate the model
best_model = grid_search.best_estimator_  # Extracts the model with best performing hyperparameters
y_pred = best_model.predict(X_test)  # Makes binary predictions (Benign (0) or Malignant (1))
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
# [:, 1] selects probability of Malignant class
# Example: 0.92 means 92% confidence in Malignant diagnosis

# Save the model in models directory
model_path = '../models/cancer_diagnosis_model.joblib'
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# %% [markdown]
# **ROC Curve**: True Positive Rate (TPR) vs False Positive Rate (FPR) at different classification thresholds.
# 
# **AUC** measures the entire area underneath the ROC curve.
# 
# **ROC-AUC Score:** The probability that a randomly chosen positive case receives a higher score than a randomly chosen negative case.

# %% [markdown]
# **Example: Cancer Diagnosis Model**
# 
# Consider a model that predicts cancer diagnosis with the following ROC-AUC scores:
# 
# `ROC-AUC = 1.0 (Perfect)`
# 
# Interpretation: If you select any patient with cancer and any patient without cancer, 100% of the time the model will give the cancer patient a higher risk score.
# 
# Example:
# 
# Patient A (has cancer): predicted risk 0.9
# Patient B (has cancer): predicted risk 0.8
# Patient C (no cancer): predicted risk 0.2
# 
# Patient D (no cancer): predicted risk 0.1
# Every cancer patient gets a higher score than every non-cancer patient
# 
# 
# ---
# 
# 
# 
# `ROC-AUC = 0.9 (Excellent)`
# 
# Interpretation: If you select any patient with cancer and any patient without cancer, 90% of the time the model will give the cancer patient a higher risk score.
# 
# Example:
# 
# Patient A (has cancer): predicted risk 0.9
# Patient B (has cancer): predicted risk 0.7
# Patient C (no cancer): predicted risk 0.8
# Patient D (no cancer): predicted risk 0.1
# 
# Most cancer patients get higher scores, but there are some overlaps
# 
# 
# ---
# 
# 
# 
# `ROC-AUC = 0.75 (Good)`
# 
# Interpretation: If you select any patient with cancer and any patient without cancer, 75% of the time the model will give the cancer patient a higher risk score.
# 
# Example:
# 
# Patient A (has cancer): predicted risk 0.8
# Patient B (has cancer): predicted risk 0.6
# Patient C (no cancer): predicted risk 0.7
# Patient D (no cancer): predicted risk 0.3
# 
# There's more overlap between the scores of positive and negative cases
# 
# 
# ---
# 
# 
# 
# `ROC-AUC = 0.5 (Random)`
# 
# Interpretation: If you select any patient with cancer and any patient without cancer, 50% of the time the model will give the cancer patient a higher risk score (equivalent to random guessing).
# 
# Example:
# 
# Patient A (has cancer): predicted risk 0.6
# Patient B (has cancer): predicted risk 0.3
# Patient C (no cancer): predicted risk 0.7
# Patient D (no cancer): predicted risk 0.2
# 
# No clear pattern in how the model ranks positive vs negative cases

# %%
# Calculate and print ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc) # 1.0 perfect, 0.5 is random (flipping a coin), less than 0.5 is worse than random.

# %%
# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save plot before showing it
plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# Random Forests calculate feature importance through a measure called "Mean Decrease in Impurity" (MDI) or "Gini importance".
# 
# At each split node, the algorithm calculates how much that split improved the purity of the data (reduced Gini impurity or entropy). This improvement is weighted by the number of samples that reach that node. Each feature gets credit for the improvements in purity for all splits where it was used.
# 
# `Feature Importance = (∑ (improvement in purity × samples at node)) / (total samples)`

# %%
# Get feature importance scores
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = X.columns

# Create DataFrame of feature importances
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values('importance', ascending=False)

# Create feature importance plot
plt.figure(figsize=(12, 6))
plt.bar(importance_df['feature'], importance_df['importance'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance in Breast Cancer Classification')
plt.tight_layout()

# Save plot before showing it
plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Print top 5 most important features
print("\nTop 5 Most Important Features:")
print(importance_df.head())

# %% [markdown]
# ## Additional Model Validation and Analysis

# %% [markdown]
# ### 1. Learning Curves Analysis

# %% [markdown]
# Learning curves analyse how an algorithm's performance changes with different amounts of training data, plotting both training and cross-validation scores. Rather than evaluating your specific tuned model, it tests the basic Random Forest algorithm by training many fresh versions with increasing data sizes. In your case, the flat training score of 1.0 and high CV score (0.985-0.99) shows that Random Forests tend to perfectly memorise training data while still generalising very well to new data on this problem. This pattern helps diagnose if your modeling approach is fundamentally sound, if you need more data, and whether you're dealing with overfitting or underfitting - all independent of the specific hyperparameters you eventually choose.

# %% [markdown]
# `train_sizes = np.linspace(0.1, 1.0, 10)` creates 10 equally spaced fractions of your training data. If you have 520 training samples (after 20% test split from 650): 0.1 = 52 samples, 0.2 = 104 samples, 0.3 = 156 samples ...and so on
# 
# For each of these sample sizes:
# 
# 1. Randomly select that many samples from training data
# 2. Use 5-fold CV to:
#   - Train model on 80% of those samples
#   - Test on remaining 20%

# %%
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y):
    """Plot learning curves to analyze bias-variance tradeoff"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc'
    )

    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')

    # Plot standard deviation bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

    plt.xlabel('Training Examples')
    plt.ylabel('ROC-AUC Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save plot before showing it
    plt.savefig('../results/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot learning curves
plot_learning_curves(pipeline, X_train, y_train) # pipeline contains scalar and classifier

# %% [markdown]
# This suggests:
# 
# Your model might be more complex than necessary and could benefit from more regularisation via:
# 
# - Reducing max_depth
# - Increasing min_samples_leaf
# - Increasing min_samples_split
# 
# However, given that:
# 
# - CV score is extremely high (>0.98)
# - Gap is relatively small (≈0.01)
# - You're working with medical data (cancer diagnosis)
# 
# This level of overfitting might be acceptable since:
# 
# - The model generalises very well despite memorising
# - In medical contexts, slightly overfitting to get better performance can be justified
# - The high CV score suggests the patterns learned are mostly valid, not just noise

# %% [markdown]
# ### 2. Handling Class Imbalance

# %% [markdown]
# The code addresses class imbalance through class weights, which penalise misclassification of the minority class more heavily.
# 
# If you have 100 majority cases and 20 minority cases:
# - Majority class weight = 100/100 = 1.0
# - Minority class weight = 100/20 = 5.0
# 
# 
# These class weights are then used in the RandomForest

# %%
# Check class distribution
class_distribution = pd.Series(y_train).value_counts()
print("\nClass Distribution in Training Set:")
print(class_distribution)

# Calculate class weights
class_weights = dict(zip(
    class_distribution.index,
    class_distribution.max() / class_distribution
))

# Create balanced pipeline using class weights
balanced_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight=class_weights,
        n_estimators=200
    ))
])

# Train and evaluate the balanced model
balanced_pipeline.fit(X_train, y_train)
y_pred_balanced = balanced_pipeline.predict(X_test)

print("\nClassification Report (Balanced Model):")
print(classification_report(y_test, y_pred_balanced, target_names=le.classes_))

# %%
# Compare ROC curves for balanced and unbalanced models
fig, ax = plt.subplots(figsize=(10, 6))

# Plot ROC curve for balanced model
y_pred_proba_balanced = balanced_pipeline.predict_proba(X_test)[:, 1]
fpr_balanced, tpr_balanced, _ = roc_curve(y_test, y_pred_proba_balanced)
roc_auc_balanced = auc(fpr_balanced, tpr_balanced)

plt.plot(fpr_balanced, tpr_balanced,
         label=f'Balanced Model (AUC = {roc_auc_balanced:.2f})',
         color='blue')

# Plot ROC curve for original model
fpr_original, tpr_original, _ = roc_curve(y_test, y_pred_proba)
roc_auc_original = auc(fpr_original, tpr_original)

plt.plot(fpr_original, tpr_original,
         label=f'Original Model (AUC = {roc_auc_original:.2f})',
         color='red')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Balanced vs Unbalanced Models')
plt.legend(loc="lower right")
plt.grid(True)

# Save plot before showing it
plt.savefig('../results/roc_balanced_vs_unbalanced.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# Marginally better performance for the balanced model.

# %%
print("First few FPR values:", fpr_balanced[:5])
print("First few TPR values:", tpr_balanced[:5])

# %% [markdown]
# TPR (True Positive Rate) jumps quickly, while FPR (False Positive Rate) stays at 0 for the first few thresholds. This means the model is very confident in its predictions as it can identify many true positives before making any false positive predictions.

# %% [markdown]
# ### 3. Feature Interaction Analysis

# %%
def plot_feature_interactions(model, X, y, le, top_n=3):
    """
    Plot interactions between top features from a model.

    Parameters:
    - model: fitted Pipeline containing RandomForestClassifier
    - X: feature DataFrame
    - y: target labels
    - le: LabelEncoder used for target encoding
    - top_n: number of top features to plot (default=3)
    """
    # Get feature importances and top features
    importances = model.named_steps['classifier'].feature_importances_
    top_features_idx = np.argsort(importances)[-top_n:]
    top_features = X.columns[top_features_idx]

    # Create figure and axes
    fig, axes = plt.subplots(top_n, top_n, figsize=(15, 15))
    fig.suptitle('Feature Interactions (Top Features)', y=1.02, fontsize=14)

    # Plot each feature pair
    for i in range(top_n):
        for j in range(top_n):
            ax = axes[i, j]
            feat_i = top_features[i]
            feat_j = top_features[j]

            # Clear default labels
            ax.set_xlabel('')
            ax.set_ylabel('')

            if i == j:  # Diagonal: histogram
                sns.histplot(
                    data=X,
                    x=feat_i,
                    hue=le.inverse_transform(y),
                    ax=ax,
                    alpha=0.6
                )
                # Rotate labels for better readability
                ax.tick_params(axis='x', rotation=45)
                # Add 'Count' label for histograms
                ax.set_ylabel('Count')

            else:  # Off-diagonal: scatter plot
                sns.scatterplot(
                    data=X,
                    x=feat_j,  # Column (j) determines x-axis
                    y=feat_i,  # Row (i) determines y-axis
                    hue=le.inverse_transform(y),
                    ax=ax,
                    alpha=0.6
                )

            # Add labels only on edges of the grid
            if i == top_n-1:  # Bottom row
                ax.set_xlabel(feat_j, fontsize=10)
            if j == 0:  # Leftmost column
                if i != j:  # Don't override 'Count' label for histograms
                    ax.set_ylabel(feat_i, fontsize=10)

            # Keep legend on all plots
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title('')  # Remove legend title
                legend.set_bbox_to_anchor((1.05, 1))  # Position legend outside plot
                legend.set_frame_on(True)  # Add frame to legend

    plt.tight_layout()
    return fig, axes

# Example usage:
fig, axes = plot_feature_interactions(best_model, X, y, le)

# Save plot before showing it
plt.savefig('../results/feature_interactions.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Histograms:** Show that these are indeed important features as they show good separation between classes considering the feature by itself.
# 
# **Scatter Plots:** Normally, strong correlation between features isn't ideal because it means you're essentially using redundant information (multicollinearity).
# However, in this case it's less of a concern because:
# 
# The separation between classes happens along the correlation line itself - benign cases cluster in the lower left, malignant in upper right
# Even though perimeter_worst and area_worst are naturally correlated (bigger perimeter = bigger area), their combination with concave_points_worst helps reinforce the class separation
# Random Forests are also generally robust to correlated features, unlike something like linear regression where it would be more problematic
# 
# What matters most here is the clear separation between classes. The correlation actually helps in this case because it shows that malignant tumors consistently show higher values across multiple related measurements, making the pattern more reliable for diagnosis.

# %% [markdown]
# ### 4. Cross-Validation with Different Metrics

# %%
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': 'roc_auc'
}

# Perform cross-validation with multiple metrics
cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)

# Print results
print("\nCross-validation results:")
for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# %% [markdown]
# ### 5. Predictions On New Data

# %%
def predict_diagnosis(new_data):
    """
    Make predictions on new data using the trained model

    Parameters:
    new_data (pd.DataFrame): DataFrame containing the same features as training data

    Returns:
    tuple: (predictions, prediction_probabilities, prediction_details)
    """
    # Get predictions and all probabilities
    predictions = best_model.predict(new_data)
    probabilities_all = best_model.predict_proba(new_data)

    # Get probability of predicted class for each case
    prediction_probabilities = np.array([prob[pred] for prob, pred in zip(probabilities_all, predictions)])

    # Create detailed prediction info
    prediction_details = pd.DataFrame({
        'Predicted_Diagnosis': le.inverse_transform(predictions),
        'Confidence': prediction_probabilities,
        'Probability_Benign': probabilities_all[:, 0],
        'Probability_Malignant': probabilities_all[:, 1]
    })

    # Add confidence level category
    prediction_details['Confidence_Level'] = pd.cut(
        prediction_details['Confidence'],
        bins=[0, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    return (
        le.inverse_transform(predictions),
        prediction_probabilities,
        prediction_details
    )

# Example of using the prediction function
new_patient = pd.DataFrame([X_test.iloc[0]])  # Example using first test case
prediction, probability, details = predict_diagnosis(new_patient)

print(f"Predicted diagnosis: {prediction[0]}")
print(f"Confidence in prediction: {probability[0]:.2f} ({details['Confidence_Level'].iloc[0]})")
print("\nDetailed Probabilities:")
print(f"Probability Benign: {details['Probability_Benign'].iloc[0]:.2f}")
print(f"Probability Malignant: {details['Probability_Malignant'].iloc[0]:.2f}")

# Example with multiple cases
multiple_patients = pd.DataFrame(X_test.iloc[0:5])
predictions, probabilities, details = predict_diagnosis(multiple_patients)

print("\nMultiple Patient Predictions:")
display(details)

# %% [markdown]
# ### 6. Model Interpretability with SHAP

# %%
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_predictions_with_shap(model, X_sample):
    """
    Analyze predictions using SHAP values with proper 3D to 2D conversion

    Parameters:
    model: trained model
    X_sample: sample data to analyze
    """
    # Get the preprocessed data using the pipeline's transform
    X_processed = model.named_steps['scaler'].transform(X_sample)
    X_processed = pd.DataFrame(X_processed, columns=X_sample.columns)

    # Get the Random Forest classifier from the pipeline
    rf_classifier = model.named_steps['classifier']

    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_classifier)

    try:
        # Get SHAP values and handle 3D array case
        shap_values = explainer.shap_values(X_processed)

        # Check if we have a 3D array and extract values for positive class (class 1)
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Extract values for positive class (index 1)
            shap_values_to_plot = shap_values[:, :, 1]
        elif isinstance(shap_values, list):
            # If we get a list of arrays, use the second one (positive class)
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        print(f"Final SHAP values shape for plotting: {shap_values_to_plot.shape}")

        # Generate summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_to_plot,
            X_processed,
            show=False
        )
        plt.title("SHAP Values Distribution")
        plt.tight_layout()
        
        # Save plot before showing it
        plt.savefig('../results/shap_values_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Calculate feature importance
        mean_abs_shap = np.abs(shap_values_to_plot).mean(0)
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': mean_abs_shap
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        print("\nTop 10 most important features based on SHAP values:")
        print(feature_importance.head(10))

        # Generate dependence plots for top 3 features
        top_features = feature_importance['feature'].head(3).tolist()
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                shap_values_to_plot,
                X_processed,
                show=False
            )
            plt.title(f"SHAP Dependence Plot for {feature}")
            plt.tight_layout()
            
            # Save plot before showing it
            plt.savefig(f'../results/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
            plt.show()
        

    except Exception as e:
        print(f"Error during SHAP analysis: {str(e)}")
        print("Shapes:")
        print(f"X_processed shape: {X_processed.shape}")
        if 'shap_values' in locals():
            if isinstance(shap_values, list):
                print("SHAP values shapes:", [arr.shape for arr in shap_values])
            else:
                print(f"SHAP values shape: {shap_values.shape}")

# Sample fewer examples to reduce computation time
X_sample = X_test[:50]  # Take smaller sample for SHAP analysis
analyze_predictions_with_shap(best_model, X_sample)

# %% [markdown]
# **SHAP Summary Plot (first plot)**: Shows how each feature impacts model predictions. Each dot represents a sample, and the colour indicates feature value (red=high, blue=low). Position on x-axis shows impact on prediction (negative=pushes towards benign, positive=pushes towards malignant). Features are ordered by importance (top=most important).
# 
# **Dependence Plots (for top 3 features):** Shows how a single feature's value affects its SHAP value. The points are coloured by another feature that interacts strongly. Strongest interaction means: which other feature most affects the relationship between the main feature and its SHAP values
# 
# The key difference from regular feature importance is that SHAP shows:
# 
# - Direction of impact (positive/negative)
# - Individual sample effects
# - Feature interactions
# - Non-linear relationships


