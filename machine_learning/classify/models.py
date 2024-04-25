from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_classification_model_grid(all_models, selected_models, seed=42):

    selected_models = [globals()[model_string] for model_string in selected_models]

    models = [(RandomForestClassifier(class_weight="balanced", random_state=seed),
               {'n_estimators': [100],
                'criterion': ['entropy', 'gini'],
                'max_depth': [10],
                'max_features': ['sqrt', 'log2']})
              ]

    # models = [(RandomForestClassifier(class_weight="balanced", random_state=seed),
    #            {'n_estimators': [100],
    #             'criterion': ['entropy', 'gini'],
    #             'max_depth': [10],
    #             'max_features': ['sqrt', 'log2']}),
    #           (GradientBoostingClassifier(random_state=seed),
    #            {'loss': ['log_loss', 'exponential'],
    #             'learning_rate': [0.01, 0.3],
    #             'n_estimators': [100],
    #             'max_depth': [10],
    #             'max_features': ['sqrt', 'log2']}),
    #           (LogisticRegression(solver='saga', max_iter=5000, class_weight="balanced", penalty='elasticnet',
    #                               random_state=seed),
    #            {'l1_ratio': [0.0, 0.5, 1.0],
    #             'C': [0.1, 5.0]}),
    #           (DummyClassifier(strategy='prior', random_state=seed), {})
    #           ]

    if all_models:
        return models
    else:
        return [(clf, params) for clf, params in models if clf.__class__ in selected_models]


def positive_class_probability(model, X):
    model_name = str(model.__class__.__name__)
    if model_name == 'Pipeline':
        model_name = str(model.named_steps['model'].__class__.__name__)
    if model_name in ['DummyClassifier', 'LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier']:
        return model.predict_proba(X)[:, 1]
    if model_name in ['SVC']:
        return model.decision_function(X)
    raise TypeError('Model name not found')


def get_feature_importance(trained_model):
    model_name = str(trained_model.__class__.__name__)
    if model_name == 'Pipeline':
        trained_model = trained_model.named_steps['model']
        model_name = str(trained_model.__class__.__name__)
    if model_name in ['RandomForestClassifier', 'GradientBoostingClassifier']:
        return trained_model.feature_importances_.flatten()
    if model_name in ['LogisticRegression'] or (model_name == 'SVC' and trained_model.kernel == 'linear'):
        return trained_model.coef_.flatten()
    return None
