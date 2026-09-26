import datetime
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from aula.core import SistemaAsistencia, CFG_DEFECTO


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = SistemaAsistencia.__new__(SistemaAsistencia)
        self.s.cfg = CFG_DEFECTO.copy()
        self.s.tz = ZoneInfo('America/Lima')
        self.s.archivo_estudiantes = str(Path(self.tmp.name) / 'estudiantes.csv')
        self.s.archivo_embeddings = str(Path(self.tmp.name) / 'embeddings.pkl')
        self.s.dir_rostros = self.tmp.name
        self.s.dir_asistencia = self.tmp.name
        self.s.cargar_datos()

    def test_reject_path_id(self):
        with self.assertRaises(ValueError):
            self.s.agregar_estudiante('../otro', 'Prueba', 'Persona')
        self.assertFalse(Path(self.s.archivo_embeddings).exists())

    def test_exact_mean_three_photos(self):
        self.s.agregar_estudiante('001', 'Prueba', 'Persona')
        vectors = np.zeros((3, 512))
        vectors[0, 0] = 1
        vectors[1, 1] = 1
        vectors[2, 1] = 1
        for vector in vectors:
            self.s._guardar_embedding('001', vector)
        expected = vectors.sum(axis=0)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(self.s.matriz[0], expected)

    def test_daily_deduplication_and_restart(self):
        self.s.agregar_estudiante('001', 'Prueba', 'Persona')
        self.assertTrue(self.s.marcar_asistencia('001', 0.8))
        self.assertFalse(self.s.marcar_asistencia('001', 0.9))
        self.s.iniciar_sesion()
        self.assertFalse(self.s.marcar_asistencia('001', 0.9))
        self.assertEqual(len(pd.read_csv(self.s.archivo_asistencia)), 1)

    def test_midnight_rollover(self):
        clock = [datetime.datetime(2026, 9, 26, 23, 59, tzinfo=self.s.tz)]
        self.s.ahora = lambda: clock[0]
        self.s.agregar_estudiante('001', 'Prueba', 'Persona')
        self.s.marcar_asistencia('001', 0.8)
        first = self.s.archivo_asistencia
        clock[0] += datetime.timedelta(minutes=2)
        self.assertTrue(self.s.marcar_asistencia('001', 0.8))
        self.assertNotEqual(first, self.s.archivo_asistencia)
        self.assertTrue(Path(first).exists())


if __name__ == '__main__':
    unittest.main()
