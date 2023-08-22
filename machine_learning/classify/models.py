from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_classification_model_grid(class_weighting=None, seed=42):
    return [(RandomForestClassifier(class_weight="balanced", random_state=seed),
             {'n_estimators': [5],
              'criterion': ['gini'],
              'max_depth': [5],
              'max_features': ['sqrt']})]
    # return [#(DummyClassifier(strategy='prior', random_state=seed), {}),
    #         (RandomForestClassifier(class_weight=class_weighting, random_state=seed),
    #          {'n_estimators': [10, 100],
    #           'criterion': ['gini', 'entropy'],
    #           'max_depth': [5, 10, None],
    #           'max_features': ['sqrt', 'log2', None]}),
    #         (GradientBoostingClassifier(random_state=seed),
    #          {'loss': ['log_loss', 'exponential'],
    #           'learning_rate': [0.01, 0.1, 0.3],
    #           'n_estimators': [100, 1000],
    #           'max_depth': [3, 5, 10],
    #           'max_features': ['sqrt', 'log2', None]}),
    #         (LogisticRegression(solver='saga', max_iter=5000, class_weight=class_weighting, penalty='elasticnet', random_state=seed),
    #          {'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
    #           'C': [0.1, 0.5, 1.0, 5.0]}),
    #         (SVC(class_weight=class_weighting, probability=True, random_state=seed),
    #          {'C': [0.1, 0.5, 1.0, 10.0, 100.0],
    #           'kernel': ['rbf', 'poly', 'sigmoid', 'linear']})
    #         ]


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
