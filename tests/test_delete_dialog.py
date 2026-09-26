import unittest
from unittest.mock import Mock, patch
from aula.local import eliminar


class DeleteDialogTests(unittest.TestCase):
    def test_cancel_confirmation_does_not_delete(self):
        sistema = Mock()
        sistema.info = {'001': {'apellidos': 'PRUEBA', 'nombres': 'Persona'}}
        with patch('aula.local.tk.Tk') as window, \
             patch('aula.local.simpledialog.askstring', return_value='001'), \
             patch('aula.local.messagebox.askyesno', return_value=False), \
             patch('aula.local.messagebox.showinfo'):
            eliminar(sistema)
        sistema.eliminar_estudiante.assert_not_called()
        window.return_value.destroy.assert_called_once()

    def test_confirm_deletes_only_selected_id(self):
        sistema = Mock()
        sistema.info = {'001': {'apellidos': 'PRUEBA', 'nombres': 'Persona'}}
        with patch('aula.local.tk.Tk'), \
             patch('aula.local.simpledialog.askstring', return_value=' 001 '), \
             patch('aula.local.messagebox.askyesno', return_value=True), \
             patch('aula.local.messagebox.showinfo'):
            eliminar(sistema)
        sistema.eliminar_estudiante.assert_called_once_with('001')
