import unittest

from load_data import load_data


class MyTestCase(unittest.TestCase):
    def test_load_data(self):
        image, roi = load_data('data/roi.xml')
        self.assertEqual(image.shape, (3, 227, 227))
        self.assertEqual(roi, [[0, 1, 2, 3], [4, 5, 6, 7]])


if __name__ == '__main__':
    unittest.main()
