from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


class ModelTrainer:

    def __init__(self, config, preprocessor, clipping_transformer):
        self.config = config
        self.preprocessor = preprocessor
        self.clipping_transformer = clipping_transformer

    def _build_cv(self):
        cv_cfg = self.config['cv']
        return StratifiedKFold(
            n_splits=cv_cfg['n_splits'],
            shuffle=cv_cfg['shuffle'],
            random_state=cv_cfg['random_state']
        )

    def _build_model(self, model_name):
        if model_name == "logistic_regression":
            return LogisticRegression(
                **self.config['models'][model_name]['fixed_params']
            )

        elif model_name == "random_forest":
            return RandomForestClassifier(
                **self.config['models'][model_name]['fixed_params']
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def simple_voting(self, model1, model2, voting="soft"):
        return VotingClassifier(
            estimators=[
                ("model1", model1),
                ("model2", model2)
            ],
            voting=voting
        )    

    def _build_sampler(self):
        sampling_cfg = self.config.get("sampling")

        if sampling_cfg is None:
            return None

        sampler_type = sampling_cfg.get("type")

        if sampler_type == "RandomUnderSampler":
            return RandomUnderSampler(
                random_state=sampling_cfg.get("random_state", 42)
            )

        raise ValueError(f"Unsupported sampler type: {sampler_type}")
    
    def _build_pipeline(self, model):
        return Pipeline([
            ('clipping', self.clipping_transformer),
            ('preprocessor', self.preprocessor),
            ('sampling', self._build_sampler()),
            ('model', model)
        ])

    def _build_param_grid(self, model_name):
        sampling_space = self.config['sampling']['search_space']

        model_space = self.config['models'][model_name]['search_space']

        param_grid = {
            'sampling__sampling_strategy': sampling_space['sampling_strategy']
        }

        for param, values in model_space.items():
            param_grid[f"model__{param}"] = values

        return [param_grid]

    def train(self, X, y, model_name):

        pipeline = self._build_pipeline(
            self._build_model(model_name)
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=self._build_param_grid(model_name),
            cv=self._build_cv(),
            scoring=self.config['grid_search']['scoring'],
            n_jobs=self.config['grid_search']['n_jobs'],
            verbose=self.config['grid_search']['verbose'],
            refit=self.config['grid_search']['refit']
        )

        grid.fit(X, y)
        return grid





    

    
   
      




