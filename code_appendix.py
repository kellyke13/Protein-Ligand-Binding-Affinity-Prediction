# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import bootstrap

# Load the training data and the public test set
X_fps_train, X_embed_train = pd.read_csv('X_fps_train.csv', index_col = 0), pd.read_csv('X_embed_train.csv', index_col = 0) # training data
y_train = pd.read_csv('y_train.csv', index_col = 0).to_numpy() # training label

X_fps_public_test, X_embed_public_test = pd.read_csv('X_fps_public_test.csv', index_col = 0), pd.read_csv('X_embed_public_test.csv', index_col = 0) # public test set
y_public_test = pd.read_csv('y_public_test.csv', index_col = 0) # public test label

X_fps_private_test, X_embed_private_test = pd.read_csv('X_fps_private_test.csv', index_col = 0), pd.read_csv('X_embed_private_test.csv', index_col = 0) # private test set 

# Define a 5-fold cross-validation
KFCV=KFold(n_splits=5, shuffle=True, random_state=214)

# Append FPS features to Embed features (column-wise)
X_combined_train = pd.concat([X_embed_train, X_fps_train], axis=1).to_numpy()
X_combined_public_test = pd.concat([X_embed_public_test, X_fps_public_test], axis=1).to_numpy()
X_combined_private_test = pd.concat([X_embed_private_test, X_fps_private_test], axis=1).to_numpy()

# Define preprocessing for embed features
embed_preprocessing = Pipeline([
    ('feature_selection', VarianceThreshold(threshold=0)),  # Remove constant features
    ('scaler', StandardScaler()),  # Normalise embed features
    ('pca', PCA()),  # PCA to reduce dimensions
    ('box-cox transform', YeoJohnsonTransformer()), # Box-Cox transformation
])

# Define preprocessing for fps features
fps_preprocessing = Pipeline([
    ('scaler', StandardScaler()),  # Normalize embed features
    ('pca', PCA()),  # PCA to Reduce dimensions
    ('box-cox transform', YeoJohnsonTransformer()), # Yeo-Johnson (Extended Box-Cox) transformation
])

# Combine preprocessing for embed and fps
combined_preprocessing = ColumnTransformer([
    ('embed_processing', embed_preprocessing, list(range(X_embed_train.shape[1]))),  # Process only for embed features
    ('fps_processing', fps_preprocessing, list(range(X_embed_train.shape[1], X_combined_train.shape[1])))  # Process only for fps features
])

# Define full pipeline
combined_elasticnet_pipeline = Pipeline([
    ('preprocessing', combined_preprocessing),  # Apply preprocessing
    ('regressor', ElasticNet())  # Elastic Net Regression
])

# Define hyperparameter grid
combined_elasticnet_param_grid = {
    'preprocessing__embed_processing__pca__n_components': [0.95, 0.98],  # PCA for embed
    'preprocessing__fps_processing__pca__n_components': [0.95, 0.98],  # PCA for fps
    'regressor__alpha': np.logspace(-1, 0, 5),  # Regularisation strength
    'regressor__l1_ratio': np.linspace(0.02, 0.1, 5)  # Balance between L1 and L2 penalty
}

# Perform GridSearchCV with cross-validation
combined_elasticnet_grid_search = GridSearchCV(
    combined_elasticnet_pipeline,
    combined_elasticnet_param_grid,
    cv=KFCV,
    scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'],
    refit='neg_mean_squared_error', # Use MSE to refit the model
    n_jobs=-1   # Using all processors to run faster
)
# Fit the model
combined_elasticnet_grid_search.fit(X_combined_train, y_train)

# Retrieve best hyperparameters and model
combined_elasticnet_best_model = combined_elasticnet_grid_search.best_estimator_

# Retrieve cross-validation results
combined_elasticnet_cv_results = pd.DataFrame(combined_elasticnet_grid_search.cv_results_)
# Top 10 cross-validation results, with scores and best chosen parameters
combined_elasticnet_cv_results.sort_values('mean_test_neg_mean_squared_error', ascending=False)[[
    'param_preprocessing__embed_processing__pca__n_components',
    'param_preprocessing__fps_processing__pca__n_components',
    'param_regressor__alpha',
    'param_regressor__l1_ratio', 
    'mean_test_neg_mean_squared_error',
    'mean_test_neg_mean_absolute_error'
    ]].head(10)

# Performance on public test dataset
combined_elastic_y_pred = combined_elasticnet_best_model.predict(X_combined_public_test)
print(f'MSE on public test: {mean_squared_error(y_public_test, combined_elastic_y_pred)}')
print(f'MAE on public test: {mean_absolute_error(y_public_test, combined_elastic_y_pred)}')

# CV MSE results
expected_mse_private = -combined_elasticnet_grid_search.best_score_
# Extract standard deviation of test MSE across folds for the best model
std_mse_private = combined_elasticnet_grid_search.cv_results_[
    'std_test_neg_mean_squared_error'
    ][combined_elasticnet_grid_search.best_index_]

# Compute 95% confidence interval
z = norm.ppf(0.975)
ci_lower = expected_mse_private - z * (std_mse_private / np.sqrt(5))
ci_upper = expected_mse_private + z * (std_mse_private / np.sqrt(5))

# Print expected MSE and confidence interval
print(f"Expected MSE on private set: {expected_mse_private:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

# Test MSE bootstrapping
# convert to 1D arrays
y_test_combined = y_public_test.squeeze()
def mse_stat(sample_indices):
    # Compute the MSE for each sample_indices
    return mean_squared_error(y_test_combined[sample_indices], combined_elastic_y_pred[sample_indices])

# Apply bootstrap (95% confidence interval)
res = bootstrap((np.arange(len(y_public_test)),), mse_stat, n_resamples=1000, method='basic', rng=412)

# Print estimated test MSE and CI
print(f"Bootstrap Estimated Test MSE: {res.bootstrap_distribution.mean():.3f}")
print(f"95% Confidence Interval using Bootstrap: ({res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f})")

# Export the predictions on the test data in csv format
private_pred = combined_elasticnet_best_model.predict(X_combined_private_test)
prediction = pd.DataFrame(private_pred, columns=['Prediction'])
prediction.index.name='Id'
prediction.to_csv('myprediction.csv') # export to csv file