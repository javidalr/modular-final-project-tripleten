# Librerias ============================================================================================

# Funciones ============================================================================================

# 1. Calcula las métricas AUC-ROC, F1-Score y Accuracy sin ajuste de hiperparámetros para evaluar los modelos

def calculate_metrics(model, features_train, target_train, features_test, target_test, model_name):

    proba_predictions_train = cross_val_predict(model, features_train, target_train, cv=5, method='predict_proba')
    positive_probabilities_train = proba_predictions_train[:, 1]
    auc_roc_train = roc_auc_score(target_train, positive_probabilities_train)
    predictions_train = model.predict(features_train)
    accuracy_train = accuracy_score(target_train, predictions_train)
    f1_train = f1_score(target_train, predictions_train)
    predictions_test = model.predict(features_test)
    auc_roc_test = roc_auc_score(target_test, predictions_test)
    accuracy_test = accuracy_score(target_test, predictions_test)
    f1_test = f1_score(target_test, predictions_test)
    results_df = pd.DataFrame({'Modelo': [model_name],
                               'AUC-ROC Entrenamiento': [auc_roc_train],
                               'AUC-ROC Prueba': [auc_roc_test],
                               'F1-Score Entrenamiento': [f1_train],
                               'F1-Score Prueba': [f1_test],
                               'Accuracy Entrenamiento': [accuracy_train],
                               'Accuracy Prueba': [accuracy_test]})    
    return results_df


# 2. Calcula las métricas AUC-ROC, F1-Score y Accuracy con ajuste de hiperparámetros para evaluar los modelos

def hyperparam_tuning_calcule_metrics(model, param_distributions, features_train, target_train, features_test, target_test, model_name):
    
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions, scoring='roc_auc', n_iter=10, cv=5, n_jobs=-1, random_state=seed)
    random_search.fit(features_train, target_train)
    best_model = random_search.best_estimator_
    proba_predictions_train = best_model.predict_proba(features_train)
    proba_predictions_test = best_model.predict_proba(features_test)
    positive_probabilities_train = proba_predictions_train[:, 1]
    positive_probabilities_test = proba_predictions_test[:, 1]
    auc_roc_train = roc_auc_score(target_train, positive_probabilities_train)
    auc_roc_test = roc_auc_score(target_test, positive_probabilities_test)
    predictions_train = best_model.predict(features_train)
    predictions_test = best_model.predict(features_test)
    accuracy_train = accuracy_score(target_train, predictions_train)
    accuracy_test = accuracy_score(target_test, predictions_test)
    f1_train = f1_score(target_train, predictions_train)
    f1_test = f1_score(target_test, predictions_test)
    print("Mejores parámetros:", random_search.best_params_)
    results_df = pd.DataFrame({'Modelo': [model_name],
                               'AUC-ROC Entrenamiento': [auc_roc_train],
                               'AUC-ROC Prueba': [auc_roc_test],
                               'F1-Score Entrenamiento': [f1_train],
                               'F1-Score Prueba': [f1_test],
                               'Accuracy Entrenamiento': [accuracy_train],
                               'Accuracy Prueba': [accuracy_test]})
    return results_df

# Models ============================================================================================

# Logistic Regression ============================================================================================

lr = LogisticRegression().fit(features_train, target_train)

lr_score = calculate_metrics(lr, features_train, target_train, features_test, target_test, 'Logistic Regression')

lr_hyper_tuning = LogisticRegression()
param_distributions = {'C': [0.1, 1, 10],
                       'penalty': ['l1', 'l2'],
                       'solver': ['liblinear', 'saga'],
                       'max_iter': [100, 200, 300],
                       'random_state': [seed]}

lr_hyper_tuning_score = hyperparam_tuning_calcule_metrics(lr_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'Logistic Regression')

# Decision Tree Classifier ============================================================================================

dtc = DecisionTreeClassifier().fit(features_train, target_train)
dtc_score = calculate_metrics(dtc, features_train, target_train, features_test, target_test, 'Decision Tree Classifier')

dtc_hyper_tuning = DecisionTreeClassifier()
param_distributions = {'criterion': ['gini', 'entropy'],
                       'max_depth': [None, 10, 20, 30],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4],
                       'max_features': ['sqrt', 'log2', None],
                       'random_state': [seed]}

dtc_hyper_tuning_score = hyperparam_tuning_calcule_metrics(dtc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'Decision Tree Classifier')

# Random Forest Classifier ============================================================================================

rfc = RandomForestClassifier().fit(features_train, target_train)
rfc_score = calculate_metrics(rfc, features_train, target_train, features_test, target_test, 'Random Forest Classifier')

rfc_hyper_tuning = RandomForestClassifier()
param_distributions = {'n_estimators': [100, 200, 300, 400, 500],
                       'criterion': ['gini', 'entropy'],
                       'max_depth': [None, 10, 20, 30, 40, 50],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4],
                       'max_features': ['sqrt', 'log2'],
                       'bootstrap': [True, False],
                       'random_state': [seed]}

rfc_hyper_tuning_score = hyperparam_tuning_calcule_metrics(rfc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'Random Forest Classifier')

# XGBoost Classifier ============================================================================================

xgbc = XGBClassifier().fit(features_train, target_train)
xgbc_score = calculate_metrics(xgbc, features_train, target_train, features_test, target_test, 'XGBoost Classifier')

xgbc_hyper_tuning = XGBClassifier()
param_distributions = {'max_depth'    : sp_randInt(5, 50),
                       'n_estimators' : sp_randInt(50, 800),    
                       'learning_rate': sp_randFloat(),    
                       'subsample'    : sp_randFloat(),
                       'random_state' : [seed]}

xgbc_hyper_tuning_score = hyperparam_tuning_calcule_metrics(xgbc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'XGBoost Classifier')

# LightGBM Classifier ============================================================================================

lgbmc = LGBMClassifier().fit(features_train, target_train)
lgbmc_score = calculate_metrics(lgbmc, features_train, target_train, features_test, target_test, 'LightGBM Classifier')

lgbmc_hyper_tuning = LGBMClassifier()
param_distributions = {'max_depth'    : sp_randInt(5, 50),    
                       'n_estimators' : sp_randInt(50, 800),    
                       'learning_rate': sp_randFloat(),    
                       'subsample'    : sp_randFloat(),
                       'random_state' : [seed]}

lgbmc_hyper_tuning_score = hyperparam_tuning_calcule_metrics(lgbmc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'LightGBM Classifier')

# CatBoost Classifier ============================================================================================

cbc = CatBoostClassifier(verbose=500).fit(features_train, target_train)
cbc_score = calculate_metrics(cbc, features_train, target_train, features_test, target_test, 'CatBoost Classifier')

cbc_hyper_tuning = CatBoostClassifier(verbose=500)

param_distributions = {'max_depth'    : sp_randInt(5, 15),    
                       'n_estimators' : sp_randInt(50, 500),    
                       'learning_rate': sp_randFloat(),    
                       'subsample'    : sp_randFloat(),
                       'random_state' : [seed]}

cbc_hyper_tuning_score = hyperparam_tuning_calcule_metrics(cbc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'CatBoost Classifier')