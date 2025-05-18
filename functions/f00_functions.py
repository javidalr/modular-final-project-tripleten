# Librerias ============================================================================================
import joblib
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV


seed = 200

# Funciones ============================================================================================

# 1.Calcula las métricas AUC-ROC, F1-Score y Accuracy sin ajuste de hiperparámetros para evaluar los modelos ============================================================================================
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

# TODO #! REVISAR RETORNOS DE FUNCIONES 

# 2. Calcula las métricas AUC-ROC, F1-Score y Accuracy con ajuste de hiperparámetros para evaluar los modelos
def hyperparam_tuning_calculate_metrics(model, param_distributions, features_train, target_train, features_test, target_test, model_name):
    
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



# Guardar de funciones ============================================================================================
joblib.dump(calculate_metrics, 'functions/f01_calculate_metrics.joblib')
joblib.dump(hyperparam_tuning_calculate_metrics, 'functions/f02_hyperparam_tuning_calculate_metrics.joblib')