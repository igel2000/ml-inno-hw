from sklearn.linear_model import(LinearRegression, # для построения моделей линейной регрессии
                                 LogisticRegression, 
                                 Ridge, 
                                 Lasso, 
                                 ElasticNet
)
from sklearn.preprocessing import PolynomialFeatures # для преобразования исходных признаков в полиномиальные, для построения моделей полиномиальной регрессии
# Импорт метрик для оценки качества моделей
from sklearn.metrics import (
    mean_squared_error,  # Средняя квадратичная ошибка для регрессии
    mean_absolute_error, 
    root_mean_squared_error, 
    r2_score  # Коэффициент детерминации для регрессии
)

from pathlib import Path, PurePath, WindowsPath, PosixPath, PureWindowsPath, PurePosixPath

def add_scores(models_scores, idx, y_train, y_train_pred, y_test, y_test_pred, params="", coef=""):
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)    
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)    
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
        
    models_scores.loc[idx, "mse_train"] = mse_train
    models_scores.loc[idx, "mse_test"] = mse_test
    models_scores.loc[idx, "rmse_train"] = rmse_train
    models_scores.loc[idx, "rmse_test"] = rmse_test
    models_scores.loc[idx, "mae_train"] = mae_train
    models_scores.loc[idx, "mae_test"] = mae_test
    models_scores.loc[idx, "r2_score_train"] = r2_train
    models_scores.loc[idx, "r2_score_test"] = r2_test
    models_scores.loc[idx, "params"] = params
    models_scores.loc[idx, "coef"] = coef
    
    #joblib.dump(models_scores, Path(result_foler, models_scores_file_name), compress=3)
    
def fit_model(models_scores, model_name, model, X_train, y_train, X_test, y_test, best_params=None):

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2_train = round(model.score(X_train, y_train),4)
    r2_test = round(model.score(X_test, y_test),4)
    mae_train = round(mean_absolute_error(y_train, y_train_pred), 4)
    mae_test = round(mean_absolute_error(y_test, y_test_pred), 4)
    params = model.get_params()
    
    models_scores.loc[model_name, 'r2_train'] = r2_train
    models_scores.loc[model_name, 'r2_test'] = r2_test
    models_scores.loc[model_name, 'mae_train'] = mae_train
    models_scores.loc[model_name, 'mae_test'] = mae_test
    if best_params is not None:
        models_scores.loc[model_name, 'best_params'] = pformat(best_params)
    
    #joblib.dump(models_scores, Path(result_foler, models_scores_file_name), compress=3)
    #joblib.dump(model, Path(result_foler, model_template_filename % model_name), compress=3)    

def check_model(model, X_train, y_train, X_test, y_test):
    """Оценить качество модели"""
    # добавить f1_score и roc_auc_score
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)    
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)    
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    
    print('Train data:')
    print(f"  MSE:    {round(mse_train,4)}")
    print(f"  RMSE:   {round(rmse_train,4)}")
    print(f"  MAE:    {round(mae_train,4)}")
    print(f"  r2:     {round(r2_train,4)}")
    print(f"  median: {round(y_train.median(),4)}")

    print('Test data:')
    print(f"  MSE:    {round(mse_test,4)}")
    print(f"  RMSE:   {round(rmse_test,4)}")
    print(f"  MAE:    {round(mae_test,4)}")
    print(f"  r2:     {round(r2_test,4)}")    
    print(f"  median: {round(y_test.median(),4)}")
