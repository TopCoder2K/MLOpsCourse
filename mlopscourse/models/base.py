from typing import Optional


class BaseModel:
    """
    Represents an interface that any model used must implement.
    """
    def __init__(self, hyperparams: Optional[dict] = None) -> None:
        self.preprocessor = None
        self.hyperparams = hyperparams
        self.model = None

    def train(self, X_train, y_train) -> None:
        raise NotImplementedError()

    def eval(self, X_test, y_test) -> None:
        raise NotImplementedError()

    def __call__(self, X_sample):
        raise NotImplementedError()
