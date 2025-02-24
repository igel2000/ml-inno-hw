#TODO: переделать на .env
dataset_foler = "/home/igel/Projects/ml/ml-inno-hw/3. Machine Learning/3.04. Регуляризация. Гребневая регрессия. Лассо регрессия/dataset/"
result_foler = "/home/igel/Projects/ml/ml-inno-hw/3. Machine Learning/3.04. Регуляризация. Гребневая регрессия. Лассо регрессия/result/"

dataset_filename_after_EDA = "01_dataset_df_after_EDA.joblib"
params_filename_after_EDA = "01_params_after_EDA.joblib"

dataset_filename_after_PrepareNans = "02_dataset_wo_nan_df_after_PrepareNans.joblib"
params_filename_after_PrepareNans = "02_params_after_PrepareNans.joblib"

dataset_filename_after_CatEncoder = "03_dataset_catencoder_df_after_CatEncoder.joblib"
params_filename_after_CatEncoder = "03_params_after_CatEncoder.joblib"

dataset_filename_after_PrepareTarget = "05_dataset_catencoder_df_after_PrepareTarget.joblib"
params_filename_after_PrepareTarget = "05_params_after_PrepareTarget.joblib"

X_train_template_filename_after_split = "07_X_train_%s_after_split.pickle"
X_test_template_filename_after_split = "07_X_test_%s_after_split.pickle"
y_train_template_filename_after_split = "07_y_train_%s_after_split.pickle"
y_test_template_filename_after_split = "07_y_test_%s_after_split.pickle"


models_scores_file_name = "models_scores.pickle"
model_template_filename = "model_%s.pickle"
