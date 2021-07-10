import inspect
from apps.ml.registry import MLRegistry
from django.test import TestCase

from apps.ml.income_classifier.random_forest import RandomForestRegressor

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "crim": 0.01501,
            "rm": 5.713,
            "dis": 3.4952
        }
        my_alg = RandomForestRegressor()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('valor propiedad' in response)
        

# add below method to MLTests class:
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "boston_housing"
        algorithm_object = RandomForestRegressor()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.10"
        algorithm_owner = "Grupo 4"
        algorithm_description = "Random Forest con pre- and post- procesamiento"
        algorithm_code = inspect.getsource(RandomForestRegressor)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)