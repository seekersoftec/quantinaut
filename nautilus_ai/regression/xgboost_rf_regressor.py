from typing import Any

from xgboost import XGBRFRegressor

from nautilus_ai.common import Logger
from nautilus_ai.regression.base_model import BaseRegressionModel
from nautilus_ai.data import NautilusAIDataKitchen
from nautilus_ai.tensorboard import TBCallback


logger = Logger(__name__)


class XGBoostRFRegressor(BaseRegressionModel):
    """
    User created prediction model. The class inherits INautilusAIModel, which
    means it has full access to all Frequency AI functionality. Typically,
    users would use this to override the common `fit()`, `train()`, or
    `predict()` methods to add their custom data handling tools or change
    various aspects of the training that cannot be configured via the
    top level config.json file.
    """

    def fit(self, data_dictionary: dict, dk: NautilusAIDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [
                (data_dictionary["test_features"], data_dictionary["test_labels"])
            ]
            eval_weights = [data_dictionary["test_weights"]]

        sample_weight = data_dictionary["train_weights"]

        xgb_model = self.get_init_model(dk.pair)

        model = XGBRFRegressor(**self.model_training_parameters)

        model.set_params(callbacks=[TBCallback(dk.data_path)])
        model.fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weights,
            xgb_model=xgb_model,
        )
        # set the callbacks to empty so that we can serialize to disk later
        model.set_params(callbacks=[])

        return model
