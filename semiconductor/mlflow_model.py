import mlflow.pyfunc

from .runtime import Bundle


class EtchModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = Bundle(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        return self.bundle.predict(model_input)
