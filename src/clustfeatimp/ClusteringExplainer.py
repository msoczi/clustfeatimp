"""This module allows to find the feature importances for clustering models."""


# Import modules
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from seaborn import heatmap
import matplotlib.pyplot as plt

from skopt import BayesSearchCV
from skopt.space import Real
from skopt.space import Integer




class ClusteringExplainer:
    """
    A class used to calculate feature importanes for clustering methods.
    
    Attributes
    ----------
    random_state : int
        Controls the random seed
    fit_hiperparams : bool
        When set to `True`, applied bayesion optimiaztion for XGBoost hyperparameters optimiaztion.
    model : XGBClassifier object
        Created XGBoost model with features importance.
    preds : array
        Numpy array with predictions for XGBoost model.
    feature_importance : dict
        Dictionary with features names and importances.
    
    Methods
    -------
    fit()
        Create XGBoost classification model and fit to (X,y) data.
    plot_importances()
        Plot feture importances.
    plot_conf_matrix()
        Plot cnfusion matrix for created model.
    """
    def __init__(self, random_state = 25, fit_hiperparams = True):
        self.random_state = random_state
        self.fit_hiperparams = fit_hiperparams
        self.model = None
        self.preds = None
        self.feature_importance = None
    
    def fit(self, X, y, n_iter=5, n_jobs=-1):
        """
        Create XGBoost classification model and fit to (X,y) data.
        
        Arguments
        ----------
        X : array-like
            The input samples. Data used for clustering.
        y : array-like
            Target values. Results of clustering.
        n_iter : int
            Numer of iterations for bayesian optimization.
        n_jobs : int
            Maximum number of concurrently running workers. 
        """
        # Calculate classes weights
        classes_weights = compute_sample_weight(class_weight='balanced', y=y)
        # 
        if self.fit_hiperparams:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=self.random_state)

            # Define parameters space for hyperparams optimization
            params_search_space = {
                'learning_rate': Real(0.01, 0.3),
                'n_estimators': Integer(10, 100),
                'min_child_weight': Integer(2, 100),
                'gamma': Real(0.0, 10.0),
                'max_depth': Integer(2, 5),
                'lambda': Real(0.0, 10.0)
            }
            # Model instance init
            model = XGBClassifier(objective='multi:softmax',
                                  random_state = self.random_state,
                                  eval_metric = ['mlogloss','merror']
                                 )
            bayesian_optimization = BayesSearchCV(estimator=model,
                                search_spaces=params_search_space,
                                n_iter = n_iter,
                                n_jobs=n_jobs,
                                cv=3,
                                verbose=0,
                                random_state=self.random_state)
            bayesian_optimization.fit(X_train, y_train)
            model = bayesian_optimization.best_estimator_
            model.early_stopping_rounds = 5
            model.fit(X,y, eval_set = [(X,y)], verbose=False, sample_weight=classes_weights)
        else:
            model = XGBClassifier(objective='multi:softmax',
                                      random_state = self.random_state,
                                      eval_metric = ['mlogloss','merror'],
                                      max_depth=2,
                                      learning_rate=0.1,
                                      n_estimators=50,
                                      early_stopping_rounds=3
                                     )
            model.fit(X,y, eval_set = [(X,y)], verbose=False, sample_weight=classes_weights)
        # Add model to attributes
        self.model = model
        # Add feature importance
        feature_importance = model.get_booster().get_score(importance_type='gain')
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        self.feature_importance = feature_importance
        # Add model predictions
        preds = model.predict(X)
        self.conf_matrix = confusion_matrix(y, preds, normalize='true')
        self.bacc = balanced_accuracy_score(y, preds)
        print(f'Model balanced accuracy: {self.bacc}')
    
    def plot_importances(self):
        """
        This method create plot with feture importances.
        """
        plt.bar(range(len(self.feature_importance)), list(self.feature_importance.values()), align='center', color=['#d7a398','#d7c398','#ccd798','#add798','#98d7a3','#98d7c3'])
        plt.xticks(range(len(self.feature_importance)), list(self.feature_importance.keys()))
        plt.xticks(rotation=90)
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()
    
    def plot_conf_matrix(self):
        """
        This method create plot with cnfusion matrix for created model.
        """
        heatmap(self.conf_matrix, annot=True, cmap="Blues")
        plt.title('Model - Confusion Matrix')
        plt.show()
