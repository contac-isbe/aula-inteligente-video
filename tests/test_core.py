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
        self.s.dir_rostros_prueba = str(Path(self.tmp.name) / 'pruebas')
        self.s.dir_fichas = str(Path(self.tmp.name) / 'fichas')
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

    def test_delete_student_preserves_others_and_history(self):
        vector = np.zeros(512)
        vector[0] = 1
        for codigo in ('001', '002'):
            carpeta = Path(self.s.agregar_estudiante(codigo, 'Prueba', 'Persona'))
            (carpeta / 'foto.jpg').write_bytes(b'test')
            self.s._guardar_embedding(codigo, vector)
        ficha = Path(self.s.dir_fichas) / '001.jpg'
        ficha.parent.mkdir()
        ficha.write_bytes(b'test')
        prueba = Path(self.s.dir_rostros_prueba) / '001'
        prueba.mkdir(parents=True)
        (prueba / 'prueba.jpg').write_bytes(b'test')
        self.s.marcar_asistencia('001', .8)
        historial = Path(self.s.archivo_asistencia).read_bytes()
        self.assertTrue(self.s.eliminar_estudiante('001'))
        self.assertNotIn('001', self.s.info)
        self.assertEqual(self.s.ids, ['002'])
        self.assertFalse((Path(self.s.dir_rostros) / '001').exists())
        self.assertFalse(ficha.exists())
        self.assertFalse(prueba.exists())
        self.assertTrue((Path(self.s.dir_rostros) / '002' / 'foto.jpg').exists())
        self.assertEqual(Path(self.s.archivo_asistencia).read_bytes(), historial)
        self.s.cargar_datos()
        self.assertNotIn('001', self.s.ids)
        self.s.agregar_estudiante('001', 'Prueba', 'Persona')
        self.s._guardar_embedding('001', vector)
        self.assertIn('001', self.s.info)
        self.assertFalse(self.s.marcar_asistencia('001', .9))

    def test_delete_missing_student_does_not_change_files(self):
        roster = Path(self.s.archivo_estudiantes).read_bytes()
        self.assertFalse(self.s.eliminar_estudiante('999'))
        self.assertEqual(Path(self.s.archivo_estudiantes).read_bytes(), roster)

    def test_delete_rejects_path_id(self):
        with self.assertRaises(ValueError):
            self.s.eliminar_estudiante('../otro')


if __name__ == '__main__':
    unittest.main()
