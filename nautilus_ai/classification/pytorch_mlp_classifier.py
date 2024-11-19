from typing import Any

import torch

from nautilus_ai.classification.base_pytorch_classifier import BasePyTorchClassifier
from nautilus_ai.data import NautilusAIDataKitchen
from nautilus_ai.torch.data_convertor import (
    DefaultPyTorchDataConvertor,
    PyTorchDataConvertor,
)
from nautilus_ai.torch.mlp_model import PyTorchMLPModel
from nautilus_ai.torch.model_trainer import PyTorchModelTrainer


class PyTorchMLPClassifier(BasePyTorchClassifier):
    """
    This class implements the fit method of INautilusAIModel.
    in the fit method we initialize the model and trainer objects.
    the only requirement from the model is to be aligned to PyTorchClassifier
    predict method that expects the model to predict a tensor of type long.

    parameters are passed via `model_training_parameters` under the freqai
    section in the config file. e.g:
    {
        ...
        "freqai": {
            ...
            "model_training_parameters" : {
                "learning_rate": 3e-4,
                "trainer_kwargs": {
                    "n_steps": 5000,
                    "batch_size": 64,
                    "n_epochs": null,
                },
                "model_kwargs": {
                    "hidden_dim": 512,
                    "dropout_percent": 0.2,
                    "n_layer": 1,
                },
            }
        }
    }
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(
            target_tensor_type=torch.long, squeeze_target_tensor=True
        )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate", 3e-4)
        self.model_kwargs: dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: dict[str, Any] = config.get("trainer_kwargs", {})

    def fit(self, data_dictionary: dict, dk: NautilusAIDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        :raises ValueError: If self.class_names is not defined in the parent class.
        """

        class_names = self.get_class_names()
        self.convert_label_column_to_int(data_dictionary, dk, class_names)
        n_features = data_dictionary["train_features"].shape[-1]
        model = PyTorchMLPModel(
            input_dim=n_features, output_dim=len(class_names), **self.model_kwargs
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # check if continual_learning is activated, and retrieve the model to continue training
        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchModelTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                model_meta_data={"class_names": class_names},
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                **self.trainer_kwargs,
            )
        trainer.fit(data_dictionary, self.splits)
        return trainer
