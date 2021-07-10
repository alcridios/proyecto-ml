# file backend/server/apps/ml/income_classifier/random_forest.py
import joblib
import pandas as pd

class RandomForestRegressor:
    def __init__(self):
        path_to_artifacts = "../../research/"
        #path_to_artifacts = "/research"
        self.values_fill_missing =  joblib.load(path_to_artifacts + "entrenamiento.joblib")
        self.encoders = joblib.load(path_to_artifacts + "codificadores.joblib")
        self.model = joblib.load(path_to_artifacts + "randomForest.joblib")

    def preprocessing(self, input_data):
        try:
            input_data = pd.DataFrame(input_data, index=[0])
            input_data.fillna(self.values_fill_missing)
            for column in [
                "crim",
                "rm",
                "dis",
            ]:
                categorical_convert = self.encoders[column]
                input_data[column] = categorical_convert.transform(input_data[column])
        except Exception as e:
            print(e)
        return input_data

    def predict(self, input_data):
        return self.model.predict(input_data)

    def postprocessing(self, input_data):
        try:
            label = input_data
        except Exception as e:
            print(e)
        return {"valor propiedad": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": print(e)}
        return prediction