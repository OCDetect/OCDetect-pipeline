# import os
# import time
# import numpy as np
# from misc import logger
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
#
#
# def random_forest_classifier(X_res, y_res, users, folder_path_plots, folder_path_results, all_subjects):
#        grid = {
#         'n_estimators': [150],
#         'criterion': ['gini'],
#         'max_depth': [None],  # None means no maximum depth
#         'min_samples_split': [2],
#         'min_samples_leaf': [5],
#         'max_features': ['log2'],
#         'bootstrap': [True],
#         'random_state': [42]
#     }
#
#     users_outer_cv = list(users.unique())
#     for test_subject in users_outer_cv:
#         X_test = X_res[users == test_subject]
#         y_test = y_res[users == test_subject]
#         X_train = X_res[users != test_subject]
#         y_train = y_res[users != test_subject]
#         users_loso = users[users != test_subject]
#
#         logo = LeaveOneGroupOut()
#         grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid, scoring='f1', cv=logo)
#
#         logger.info("Grid search cv running....")
#         start_time = time.time()
#         grid_search.fit(X_train, np.squeeze(y_train), groups=np.squeeze(users_loso))
#         end_time = time.time()
#         logger.info("Grid search completed")
#
#         training_time_seconds = end_time - start_time
#         training_time_minutes = training_time_seconds / 60
#
#         # Access the best hyperparameters and model performance
#         best_params = grid_search.best_params_
#         best_score = grid_search.best_score_
#         best_model = grid_search.best_estimator_
#
#         y_pred = best_model.predict(X_test)
#         # y_pred_proba = best_model.predict_proba(X_test)[:,1]
#         f1_test = f1_score(y_test, y_pred)
#         precision_test = precision_score(y_test, y_pred)
#         recall_test = recall_score(y_test, y_pred)
#         roc_auc_test = roc_auc_score(y_test, y_pred)
#         report = classification_report(y_test, y_pred)
#
#         logger.info("Writing results to file")
#         os.makedirs(f"{folder_path_results}", exist_ok=True)
#         os.makedirs(f"{folder_path_plots}", exist_ok=True)
#
#         with open(f'{folder_path_results}output_oversampling.txt', 'a') as f:
#             f.write("\n")
#             f.write(f"test subject: {test_subject}")
#             f.write("\n")
#             f.write(f"score: {best_score}")
#             f.write("\n")
#             f.write(f"param: {best_params}")
#             f.write("\n")
#             f.write(f"f1: {f1_test}; precision: {precision_test}; recall: {recall_test}; roc_auc: {roc_auc_test};")
#             f.write("\n")
#             f.write(f"report: {report}")
#             f.write("\n")
#             f.write(f"Training time: {training_time_seconds:.2f} seconds ({training_time_minutes:.2f} minutes)")
#             f.write("\n")
#
#         # Create a confusion matrix
#         subjects = "all subjects" if all_subjects else "ocd subjects only"
#         title = f"({subjects}) Results: f1: {f1_test}; precision: {precision_test}; recall: {recall_test}; " \
#                 f"test_subject: {test_subject}"
#         make_cm(y_test, y_pred, folder_path_plots, title, test_subject)
#
#     return best_model, best_params, best_score
