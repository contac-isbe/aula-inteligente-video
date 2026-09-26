import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from aula.local import abrir_camara


class CameraTests(unittest.TestCase):
    def test_media_foundation_receives_priority(self):
        image = np.ones((10, 10, 3), dtype=np.uint8)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, image)
        with patch('aula.local.cv2.VideoCapture', return_value=cap) as create:
            self.assertIs(abrir_camara(0), cap)
            create.assert_called_once_with(0, cv2.CAP_MSMF)

    def test_failed_capture_released_before_fallback(self):
        failed = Mock()
        failed.isOpened.return_value = True
        failed.read.return_value = (False, None)
        working = Mock()
        working.isOpened.return_value = True
        working.read.return_value = (True, np.ones((10, 10, 3), dtype=np.uint8))
        with patch('aula.local.cv2.VideoCapture', side_effect=[failed, working]):
            self.assertIs(abrir_camara(0), working)
        failed.release.assert_called_once()
        working.release.assert_not_called()
