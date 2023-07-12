from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from sklearn.model_selection import GridSearchCV


def random_forest(X_res, y_res, users):

    # GridSearch
    grid = {
        'n_estimators': [100],
        'max_features': ['sqrt'],
        'criterion': ['gini'],
        'class_weight': ['balanced'],
        'random_state': [18]
    }

    # Create an instance of LeaveOneGroupOut for LOGO-CV
    logo = LeaveOneGroupOut()

    # Create an instance of GridSearchCV with LOGO-CV
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid, scoring='f1', cv=logo)

    # Fit the grid search to your dataset
    grid_search.fit(X_res, np.squeeze(y_res), groups=np.squeeze(users))

    # Access the best hyperparameters and model performance
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    return best_model, best_params, best_score
