import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(objective='reg:squarederror')

# 网格搜索参数
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)




pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', None)        # 自动换行


results = pd.DataFrame(grid_search.cv_results_)

print(results[['param_learning_rate', 'param_max_depth', 'param_n_estimators', 'mean_test_score', 'std_test_score']])
print("Best parameters:", grid_search.best_params_)
