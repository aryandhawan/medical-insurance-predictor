import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
data = pd.read_csv('insurance.csv')
df = pd.DataFrame(data=data)

categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']

# Map smoker to binary flag
df['smoker_flag'] = df['smoker'].map({'yes': 1, 'no': 0})
# Note: You can include 'smoker_flag' in numerical_cols if you want to use it as a feature
# Otherwise, remove this mapping and use original columns

# Define X (features) and y (target)
X = df.drop(columns=['charges'])
y = df['charges']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('The data has been split into train and test sets!')
# Preprocessing pipelines
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])
preprocessor_pipeline = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Model pipelines
linear_pipeline = Pipeline([
    ('preprocessor', preprocessor_pipeline),
    ('model', LinearRegression())
])

forest_pipeline = Pipeline([
    ('preprocessor', preprocessor_pipeline),
    ('model', RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))
])

tree_pipeline = Pipeline([
    ('preprocessor', preprocessor_pipeline),
    ('model', DecisionTreeRegressor(random_state=42))
])

models = {
    'Linear_model': linear_pipeline,
    'Forest_model': forest_pipeline,
    'Tree_model': tree_pipeline
}

model_results = {}
for model_name, pipeline in models.items():
    print(f'Training {model_name}')
    pipeline.fit(X_train, y_train)
    model_results[model_name] = pipeline
    print(f'{model_name} training complete!')

# Evaluation on test set
print('----------Evaluating all models on test set------------')
evaluation_results = {}
for model_name, pipeline in model_results.items():
    y_pred_test = pipeline.predict(X_test) # the sample input data is fed to the pipeline for prediction ! 
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    evaluation_results[model_name] = {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    print(f"{model_name} Test RMSE: ${test_rmse:.2f}")
    print(f"{model_name} Test MAE: ${test_mae:.2f}")
    print(f"{model_name} Test R²: {test_r2:.3f}\n")

# Cross-validation (using full X and y for demonstration, or use X_train, y_train for realistic scenario)
print('Cross validation Analysis')
print("Performing 5-fold cross validation for robust performance estimates...")
cv_results = {}
for model_name, pipeline in models.items():
    print(f'Cross validating {model_name}')
    cv_mse_scores = -cross_val_score(pipeline, X_train, y_train, 
                                    scoring='neg_mean_squared_error', cv=5)
    cv_rmse_scores = np.sqrt(cv_mse_scores)
    cv_mean = cv_rmse_scores.mean()
    cv_std = cv_rmse_scores.std()
    cv_results[model_name] = {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_scores': cv_rmse_scores
    }
    print(f"CV RMSE: {cv_mean:.2f} (+/- {cv_std * 2:.2f})")
    print(f"Individual fold scores: {cv_rmse_scores}\n")

# Model comparison table
print("------------Model comparison Table----------")
comparison_data = []
for model_name in evaluation_results.keys():
    comparison_data.append({
        'model': model_name,
        'Test RMSE': f"{evaluation_results[model_name]['test_rmse']:.3f}",
        'Test MAE': f"{evaluation_results[model_name]['test_mae']:.3f}",
        'Test R²': f"{evaluation_results[model_name]['test_r2']:.3f}",
        'CV RMSE (Mean)': f"{cv_results[model_name]['cv_mean']:.3f}",
        'CV RMSE (Std)': f"{cv_results[model_name]['cv_std']:.3f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

best_model_name = min(evaluation_results.keys(), 
                     key=lambda x: evaluation_results[x]['test_rmse'])
print(f'\nThe best model for this dataset is {best_model_name}')