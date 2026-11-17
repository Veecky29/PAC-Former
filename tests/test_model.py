import unittest
import torch
from pacformer.model import create_model

class ModelTests(unittest.TestCase):

    def test_shapes_and_initialization(self):
        config = dict(learning_rate=0.0001, d_model=128, kernel_size=5)
        states = {}
        for variant in ['B', 'P', 'S', 'F']:
            torch.manual_seed(2026)
            model = create_model(7, 96, config, variant=variant, device='cpu')
            states[variant] = model.state_dict()
            model.eval()
            with torch.no_grad():
                prediction = model(torch.randn(2, 96, 7))
            self.assertEqual(tuple(prediction.shape), (2, 96, 7))
            self.assertTrue(torch.isfinite(prediction).all())
        for variant in ['B', 'P', 'S']:
            for name, value in states[variant].items():
                self.assertTrue(torch.equal(value, states['F'][name]), name)
        for name, value in states['F'].items():
            if name.endswith('scale_factor'):
                self.assertEqual(value.item(), 0.0)
if __name__ == '__main__':
    unittest.main()
