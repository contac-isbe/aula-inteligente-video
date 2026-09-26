"""Interfaz local de video; ejecutar con python -m aula.local."""
import argparse
from collections import Counter
from pathlib import Path
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

import cv2
from PIL import Image

from aula.core import SistemaAsistencia


def abrir_camara(indice, metodo='auto'):
    metodos = {
        'auto': (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY),
        'msmf': (cv2.CAP_MSMF,),
        'dshow': (cv2.CAP_DSHOW,),
    }
    for backend in metodos[metodo]:
        cap = cv2.VideoCapture(indice, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
    raise RuntimeError(f'No se puede abrir la cámara {indice}. Prueba otro índice o cierra otras aplicaciones que usen la cámara.')


def registrar(sistema, frame):
    """La captura explícita evita registrar a un desconocido sin revisar sus datos."""
    root = tk.Tk()
    root.withdraw()
    try:
        r = sistema.analizar_foto(frame)
        if r['n_rostros'] != 1:
            messagebox.showinfo('Registro', 'Debe aparecer exactamente un rostro. Inténtalo de nuevo.', parent=root)
            return
        if r['id_est']:
            inf = sistema.info[r['id_est']]
            messagebox.showinfo('Alumno conocido', f"{r['id_est']}: {inf['apellidos']}, {inf['nombres']}", parent=root)
            return
        codigo = simpledialog.askstring('Registro', 'Código del estudiante:', parent=root)
        if not codigo:
            return
        codigo = codigo.strip()
        if codigo in sistema.info:
            inf = sistema.info[codigo]
            if messagebox.askyesno('Añadir foto', f"¿Esta persona es {inf['apellidos']}, {inf['nombres']}?", parent=root):
                sistema.agregar_foto(codigo, frame, r['emb'])
            return
        apellidos = simpledialog.askstring('Registro', 'Apellidos:', parent=root)
        if not apellidos or not apellidos.strip():
            return
        nombres = simpledialog.askstring('Registro', 'Nombres:', parent=root)
        if not nombres or not nombres.strip():
            return
        if messagebox.askyesno('Confirmar', f'{codigo}: {apellidos.strip()}, {nombres.strip()}\n¿Guardar registro?', parent=root):
            sistema.registrar_nuevo(codigo, apellidos, nombres, frame, r['emb'])
            # La asistencia se confirma después por video; no se inventa una similitud 1.0.
    except (ValueError, OSError) as exc:
        messagebox.showerror('Registro', str(exc), parent=root)
    finally:
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description='Video local del aula inteligente')
    parser.add_argument('--camara', type=int, default=0, help='Índice: 0 suele ser integrada, 1 suele ser USB. Depende del equipo.')
    parser.add_argument('--datos', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--umbral', type=float, default=0.60)
    parser.add_argument('--solo-video', action='store_true', help='Comprobar cámara sin cargar modelos faciales')
    parser.add_argument('--backend', choices=('auto', 'msmf', 'dshow'), default='auto',
                        help='Método de captura de Windows; auto prueba Media Foundation primero')
    args = parser.parse_args()
    if not 0 < args.umbral <= 1:
        parser.error('--umbral debe estar entre 0 y 1')
    cap = None
    sistema = None
    try:
        if not args.solo_video:
            print('Cargando reconocimiento facial; la primera ejecución descarga pesos del modelo.')
            sistema = SistemaAsistencia(str(args.datos), cfg={'umbral_similitud': args.umbral})
            sistema.iniciar_sesion()
        cap = abrir_camara(args.camara, args.backend)
        votos = Counter()
        anterior = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError('Se perdió la conexión con la cámara.')
            ahora = time.perf_counter()
            fps = 1 / max(ahora - anterior, 1e-6)
            anterior = ahora
            if sistema:
                sistema.asegurar_sesion()
                rostros = sistema.rostros_en_imagen(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                caras = []
                ids_actuales = set()
                for caja, prob, emb in rostros:
                    codigo, similitud = sistema.identificar(emb)
                    if codigo:
                        ids_actuales.add(codigo)
                    caras.append({'caja': caja, 'id_est': codigo, 'sim': similitud})
                # Tres frames consecutivos por identidad; no sustituye al tracking de personas.
                votos = Counter({codigo: votos[codigo] + 1 for codigo in ids_actuales})
                for cara in caras:
                    if cara['id_est'] and votos[cara['id_est']] >= 3:
                        sistema.marcar_asistencia(cara['id_est'], cara['sim'])
                lienzo = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
                import numpy as np
                salida = cv2.cvtColor(np.array(sistema.dibujar(lienzo, [], caras).convert('RGB')), cv2.COLOR_RGB2BGR)
            else:
                salida = frame.copy()
            cv2.putText(salida, f'Camara {args.camara} | {fps:.1f} FPS | R: registrar | Q: salir',
                        (8, salida.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Aula inteligente - Video', salida)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty('Aula inteligente - Video', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('r') and sistema:
                registrar(sistema, frame.copy())
                votos.clear()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        import sys
        print(f'No se pudo iniciar o continuar el video: {exc}', file=sys.stderr)
        sys.exit(1)
