import unittest
import numpy as np
from src.data_prep import load_wine_data, prepare_data
from src.model import create_model, compile_model

class TestWineQualityModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        cls.data = load_wine_data()
        cls.train_x, cls.train_y, cls.valid_x, cls.valid_y, cls.test_x, cls.test_y, cls.signature = prepare_data(cls.data)
    
    def test_data_loading(self):
        """Test if data is loaded correctly."""
        self.assertIsNotNone(self.data)
        self.assertTrue(len(self.data) > 0)
        self.assertEqual(len(self.data.columns), 12)  # 11 features + 1 target
    
    def test_data_preparation(self):
        """Test if data preparation produces correct shapes."""
        self.assertEqual(len(self.train_x.shape), 2)
        self.assertEqual(self.train_x.shape[1], 11)  # 11 features
        self.assertEqual(len(self.train_y.shape), 1)
    
    def test_model_creation(self):
        """Test if model is created with correct architecture."""
        mean = np.mean(self.train_x, axis=0)
        var = np.var(self.train_x, axis=0)
        model = create_model(self.train_x.shape[1:], mean, var)
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 7)  # Input, Normalization, Dense, Dropout, Dense, Dropout, Dense
    
    def test_model_compilation(self):
        """Test if model compiles successfully."""
        mean = np.mean(self.train_x, axis=0)
        var = np.var(self.train_x, axis=0)
        model = create_model(self.train_x.shape[1:], mean, var)
        compile_model(model, learning_rate=0.01, momentum=0.9)
        
        self.assertTrue(model.optimizer is not None)
        self.assertTrue(model.loss is not None)

if __name__ == '__main__':
    unittest.main() 