# encoding: UTF-8

import unittest
import load_data


class MyTestCase(unittest.TestCase):
    def test_load_data(self):
        folder, image, roi = load_data.parse_roi_xml('data/roi.xml')
        self.assertEqual(folder, 'test')
        self.assertEqual(image, 'data/1000.jpg')
        self.assertEqual(roi, [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_load_image(self):
        image = load_data.load_image('data/1000.jpg')
        self.assertEqual(image.shape, (3, 227, 227))


if __name__ == '__main__':
    unittest.main()
