from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from misc import logger


def get_classification_model_grid(all_models, selected_models, settings: dict, seed=42):

    selected_models = [globals()[model_string] for model_string in selected_models]

    grid_search = settings.get("grid_search")
    if not grid_search:
        models = [(RandomForestClassifier(class_weight="balanced", random_state=seed),
                   {'n_estimators': [100],
                    'criterion': ['entropy', 'gini'],
                    'max_depth': [10],
                    'max_features': ['sqrt', 'log2']}),
                  (GradientBoostingClassifier(random_state=seed),
                        {'loss': ['exponential'],
                         'learning_rate': [0.01],
                         'n_estimators': [100],
                         'max_depth': [10],
                         'max_features': ['sqrt']}),
                  (LogisticRegression(solver='saga', max_iter=5000, class_weight="balanced", penalty='elasticnet',
                                      random_state=seed),
                   {'l1_ratio': [0.0, 0.5, 1.0],
                    'C': [0.1, 5.0]}),
                  (DummyClassifier(strategy='prior', random_state=seed), {})
                  ]
    else:
      
        if settings.get("label_type") != "null_vs_rout_vs_comp":
            models = [(RandomForestClassifier(class_weight="balanced", random_state=seed),
                       {'n_estimators': [100],
                        'criterion': ['entropy', 'gini'],
                        'max_depth': [10],
                        'max_features': ['sqrt', 'log2']}),
                      (GradientBoostingClassifier(random_state=seed),
                       {'loss': ['log_loss', 'exponential'],
                        'learning_rate': [0.01, 0.3],
                        'n_estimators': [100],
                        'max_depth': [10],
                        'max_features': ['sqrt', 'log2']}),
                      (LogisticRegression(solver='saga', max_iter=5000, class_weight="balanced", penalty='elasticnet',
                                          random_state=seed),
                       {'l1_ratio': [0.0, 0.5, 1.0],
                        'C': [0.1, 5.0]}),
                      (DummyClassifier(strategy='prior', random_state=seed), {})
                      ]
        else:
            logger.info("Not using GradientBoostingClassifier for Multiclass Classification")
            models = [(RandomForestClassifier(class_weight="balanced", random_state=seed),
                       {'n_estimators': [100],
                        'criterion': ['entropy', 'gini'],
                        'max_depth': [10],
                        'max_features': ['sqrt', 'log2']}),
                      (GradientBoostingClassifier(random_state=seed),
                       {'loss': ['log_loss'],
                        'learning_rate': [0.01, 0.3],
                        'n_estimators': [100],
                        'max_depth': [10],
                        'max_features': ['sqrt', 'log2']}),
                      (LogisticRegression(solver='saga', max_iter=10000, class_weight="balanced", penalty='elasticnet',
                                          random_state=seed, multi_class='multinomial'),
                       {'l1_ratio': [0.0, 0.5, 1.0],
                        'C': [0.1, 5.0]}),
                      (DummyClassifier(strategy='prior', random_state=seed), {})
                      ]
    if all_models:
        return models
    else:
        print([(clf, params) for clf, params in models if clf.__class__ in selected_models])
        return [(clf, params) for clf, params in models if clf.__class__ in selected_models]


def positive_class_probability(model, X, binary_classification = True):
    model_name = str(model.__class__.__name__)
    if model_name == 'Pipeline':
        model_name = str(model.named_steps['model'].__class__.__name__)
    if model_name in ['DummyClassifier', 'LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier']:
        return model.predict_proba(X)[:, 1] if binary_classification else model.predict_proba(X)
    if model_name in ['SVC']:
        return model.decision_function(X)
    raise TypeError('Model name not found')


def get_feature_importance(trained_model, binary_classification):
    model_name = str(trained_model.__class__.__name__)
    if model_name == 'Pipeline':
        trained_model = trained_model.named_steps['model']
        model_name = str(trained_model.__class__.__name__)
    if model_name in ['RandomForestClassifier', 'GradientBoostingClassifier']:
        return trained_model.feature_importances_.flatten()
    if model_name in ['LogisticRegression'] or (model_name == 'SVC' and trained_model.kernel == 'linear'):
        if binary_classification:
            return trained_model.coef_.flatten()
        else:
            return trained_model.coef_
    return None
