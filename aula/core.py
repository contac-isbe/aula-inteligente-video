import os, pickle, datetime, re
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1

EXT_IMG = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def guardar_imagen(ruta, img_bgr):
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(ruta)
    return ok


_FUENTE = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
_FUENTE_B = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')

CFG_DEFECTO = {
    'umbral_similitud': 0.60,         # similitud coseno mínima para reconocer (calibrar con la sección de métricas)
    'prob_min_rostro': 0.90,          # probabilidad mínima de MTCNN
    'tam_min_rostro': 40,             # px; con la laptop el alumno está cerca, rostros grandes
    'zona_horaria': 'America/Lima',   # Colab corre en UTC: sin esto la hora saldría 5 h adelantada
}


class SistemaAsistencia:
    """Aula Inteligente: reconocimiento facial y registro de asistencia."""

    def __init__(self, base_dir, cfg=None, device=None):
        self.cfg = {**CFG_DEFECTO, **(cfg or {})}
        self.tz = ZoneInfo(self.cfg['zona_horaria'])
        self.base = base_dir
        self.dir_rostros = os.path.join(base_dir, 'rostros')                 # rostros/<ID>/*.jpg
        self.dir_rostros_prueba = os.path.join(base_dir, 'rostros_prueba')   # para métricas
        self.dir_asistencia = os.path.join(base_dir, 'asistencia')
        self.dir_sesiones = os.path.join(base_dir, 'sesiones')
        self.dir_fichas = os.path.join(base_dir, 'fichas')                   # foto + datos del alumno
        self.archivo_estudiantes = os.path.join(base_dir, 'estudiantes.csv')
        self.archivo_embeddings = os.path.join(base_dir, 'embeddings_rostros.pkl')
        for d in [self.dir_rostros, self.dir_rostros_prueba, self.dir_asistencia, self.dir_sesiones, self.dir_fichas]:
            os.makedirs(d, exist_ok=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True,
                           min_face_size=self.cfg['tam_min_rostro'], device=self.device)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.f_normal = ImageFont.truetype(_FUENTE, 15)
        self.f_bold = ImageFont.truetype(_FUENTE_B, 16)
        self.cargar_datos()

    def ahora(self):
        return datetime.datetime.now(self.tz)

    # ------------------------------------------------------------ estudiantes
    def leer_estudiantes(self):
        if not os.path.exists(self.archivo_estudiantes):
            pd.DataFrame(columns=['id_estudiante', 'apellidos', 'nombres', 'fecha_registro']).to_csv(
                self.archivo_estudiantes, index=False)
        return pd.read_csv(self.archivo_estudiantes, dtype=str).fillna('')

    def agregar_estudiante(self, id_estudiante, apellidos, nombres):
        id_estudiante = str(id_estudiante).strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', id_estudiante):
            raise ValueError('El ID debe contener solo letras, números, guion o guion bajo (1-64).')
        df = self.leer_estudiantes()
        df = df[df['id_estudiante'] != id_estudiante]
        nuevo = pd.DataFrame([{'id_estudiante': id_estudiante, 'apellidos': apellidos.strip().upper(),
                               'nombres': nombres.strip().title(),
                               'fecha_registro': self.ahora().strftime('%Y-%m-%d %H:%M:%S')}])
        pd.concat([df, nuevo], ignore_index=True).sort_values('apellidos').to_csv(self.archivo_estudiantes, index=False)
        carpeta = os.path.join(self.dir_rostros, id_estudiante)
        os.makedirs(carpeta, exist_ok=True)
        return carpeta

    # ------------------------------------------------------------ rostros
    @torch.no_grad()
    def rostros_en_imagen(self, img_rgb):
        """Detecta TODOS los rostros (MTCNN) y devuelve [(caja, prob, embedding_512)] (FaceNet)."""
        pil = Image.fromarray(img_rgb)
        cajas, probs = self.mtcnn.detect(pil)
        if cajas is None:
            return []
        ok = [k for k, p in enumerate(probs) if p is not None and p >= self.cfg['prob_min_rostro']]
        if not ok:
            return []
        cajas, probs = cajas[ok], probs[ok]
        caras = self.mtcnn.extract(pil, cajas, None)
        if caras is None:
            return []
        if caras.dim() == 3:
            caras = caras.unsqueeze(0)
        embs = torch.nn.functional.normalize(self.facenet(caras.to(self.device)), dim=1).cpu().numpy()
        return [(cajas[k].astype(int), float(probs[k]), embs[k]) for k in range(len(cajas))]

    def cargar_datos(self):
        est = self.leer_estudiantes()
        self.info = {r.id_estudiante: {'apellidos': r.apellidos, 'nombres': r.nombres} for r in est.itertuples()}
        base = {}
        if os.path.exists(self.archivo_embeddings):
            with open(self.archivo_embeddings, 'rb') as fh:
                base = pickle.load(fh)
        self.ids = [i for i in base if i in self.info]
        self.matriz = np.stack([base[i]['embedding'] for i in self.ids]) if self.ids else np.zeros((0, 512))

    def identificar(self, emb):
        """Similitud coseno contra la base. Devuelve (id_estudiante o None, similitud)."""
        if len(self.ids) == 0:
            return None, 0.0
        s = self.matriz @ emb
        k = int(np.argmax(s))
        return (self.ids[k], float(s[k])) if s[k] >= self.cfg['umbral_similitud'] else (None, float(s[k]))

    # ------------------------------------------------------------ registro individual (foto + teclado)
    def analizar_foto(self, img_bgr):
        """Busca el rostro principal de la foto y decide si el alumno ya está registrado o es nuevo."""
        rostros = self.rostros_en_imagen(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not rostros:
            return {'estado': 'sin_rostro', 'n_rostros': 0}
        caja, prob, emb = max(rostros, key=lambda r: (r[0][2] - r[0][0]) * (r[0][3] - r[0][1]))
        id_est, sim = self.identificar(emb)
        return {'estado': 'registrado' if id_est else 'nuevo', 'caja': caja, 'emb': emb,
                'id_est': id_est, 'sim': sim, 'n_rostros': len(rostros)}

    def _guardar_embedding(self, id_est, emb):
        """Agrega el embedding a la base (promedio acumulado) sin recalcular a los demás estudiantes."""
        base = {}
        if os.path.exists(self.archivo_embeddings):
            with open(self.archivo_embeddings, 'rb') as fh:
                base = pickle.load(fh)
        if id_est in base:
            n = base[id_est].get('n_imagenes', 1)
            if 'suma' not in base[id_est]:
                raise ValueError('Base antigua sin suma: vuelva a registrar las fotos para calcular el promedio exacto.')
            prom = base[id_est]['suma'] + emb
            base[id_est] = {'embedding': prom / np.linalg.norm(prom), 'suma': prom, 'n_imagenes': n + 1}
        else:
            base[id_est] = {'embedding': emb / np.linalg.norm(emb), 'suma': emb.copy(), 'n_imagenes': 1}
        with open(self.archivo_embeddings, 'wb') as fh:
            pickle.dump(base, fh)
        self.cargar_datos()

    def _guardar_foto(self, id_est, img_bgr):
        inf = self.info[id_est]
        carpeta = os.path.join(self.dir_rostros, id_est)
        os.makedirs(carpeta, exist_ok=True)
        n = len([f for f in os.listdir(carpeta) if f.lower().endswith(EXT_IMG)]) + 1
        nombre = f"{id_est}_{inf['apellidos']}_{inf['nombres']}_{self.ahora().strftime('%Y%m%d_%H%M%S')}_{n:02d}"
        nombre = '_'.join(nombre.split())                      # sin espacios en el nombre del archivo
        ruta = os.path.join(carpeta, nombre + '.jpg')
        if not guardar_imagen(ruta, img_bgr):
            raise OSError('No se pudo guardar la fotografía.')
        return ruta

    def registrar_nuevo(self, id_est, apellidos, nombres, img_bgr, emb):
        """Alumno nuevo: guarda sus datos (estudiantes.csv), su foto (rostros/<ID>/) y su ficha (fichas/<ID>.jpg)."""
        id_est = str(id_est).strip()
        self.agregar_estudiante(id_est, apellidos, nombres)
        self.cargar_datos()
        ruta_foto = self._guardar_foto(id_est, img_bgr)
        self._guardar_embedding(id_est, emb)
        ruta_ficha = self.generar_ficha(id_est, img_bgr)
        inf = self.info[id_est]
        print(f"Estudiante registrado: {id_est} – {inf['apellidos']}, {inf['nombres']}")
        return ruta_foto, ruta_ficha

    def agregar_foto(self, id_est, img_bgr, emb):
        """Alumno ya registrado que no fue reconocido: se añade la foto para mejorar su reconocimiento."""
        ruta = self._guardar_foto(id_est, img_bgr)
        self._guardar_embedding(id_est, emb)
        return ruta

    def generar_ficha(self, id_est, img_bgr):
        """Imagen de la foto con los datos ingresados por teclado impresos debajo."""
        inf = self.leer_estudiantes().set_index('id_estudiante').loc[id_est]
        foto = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        foto.thumbnail((480, 480))
        W, H = max(foto.size[0], 480), foto.size[1] + 150
        ficha = Image.new('RGB', (W, H), (255, 255, 255))
        ficha.paste(foto, ((W - foto.size[0]) // 2, 0))
        d = ImageDraw.Draw(ficha)
        y = foto.size[1]
        d.rectangle([0, y, W, y + 30], fill=(31, 56, 100))
        d.text((10, y + 6), 'FICHA DE REGISTRO – AULA INTELIGENTE (UPAO)', font=self.f_bold, fill=(255, 255, 255))
        datos = [('ID', id_est), ('Apellidos', inf['apellidos']), ('Nombres', inf['nombres']),
                 ('Fecha de registro', inf.get('fecha_registro', ''))]
        for k, (et, val) in enumerate(datos):
            d.text((10, y + 38 + 27 * k), f'{et}: {val}', font=self.f_bold if k < 3 else self.f_normal, fill=(0, 0, 0))
        ruta = os.path.join(self.dir_fichas, f'{id_est}.jpg')
        ficha.save(ruta, quality=92)
        return ruta

    def asegurar_sesion(self):
        """Abre la sesión del día si aún no existe (para marcar asistencia fuera del modo en vivo)."""
        if getattr(self, 'fecha', None) != self.ahora().strftime('%Y-%m-%d'):
            self.iniciar_sesion()

    # ------------------------------------------------------------ sesión
    def iniciar_sesion(self):
        self.cargar_datos()
        self.fecha = self.ahora().strftime('%Y-%m-%d')
        self.archivo_asistencia = os.path.join(self.dir_asistencia, f'asistencia_{self.fecha}.csv')
        self.asistencia = {}
        if os.path.exists(self.archivo_asistencia):         # si se reinicia la sesión el mismo día, no duplica
            for r in pd.read_csv(self.archivo_asistencia, dtype=str).to_dict('records'):
                self.asistencia[r['id_estudiante']] = r
        self.ultimo_evento = 'Esperando el ingreso de estudiantes...'
        print(f'Sesión {self.fecha} iniciada | estudiantes con rostro registrado: {len(self.ids)} | '
              f'asistencias previas hoy: {len(self.asistencia)}')

    def marcar_asistencia(self, id_est, sim):
        self.asegurar_sesion()
        if id_est in self.asistencia:
            return False
        t = self.ahora()
        fila = {'id_estudiante': id_est, 'apellidos': self.info[id_est]['apellidos'],
                'nombres': self.info[id_est]['nombres'], 'fecha': t.strftime('%Y-%m-%d'),
                'hora_ingreso': t.strftime('%H:%M:%S'), 'similitud': round(sim, 3), 'estado': 'Presente'}
        self.asistencia[id_est] = fila
        pd.DataFrame([fila]).to_csv(self.archivo_asistencia, mode='a', index=False,
                                    header=not os.path.exists(self.archivo_asistencia))
        self.ultimo_evento = f"✔ {fila['apellidos']}, {fila['nombres']} – ingreso {fila['hora_ingreso']}"
        print(f"[{fila['hora_ingreso']}] Asistencia registrada: {id_est} {fila['apellidos']}, {fila['nombres']}")
        return True

    def dibujar(self, lienzo, personas, caras):
        """Dibuja sobre una imagen PIL (RGB para ventana local, RGBA transparente para el overlay de Colab)."""
        d = ImageDraw.Draw(lienzo)
        W = lienzo.size[0]
        for (x1, y1, x2, y2), tid in personas:
            d.rectangle([x1, y1, x2, y2], outline=(80, 160, 255, 255), width=2)
            d.text((x1 + 4, y1 + 2), f'Persona #{tid}' if tid >= 0 else 'Persona', font=self.f_normal,
                   fill=(80, 160, 255, 255))
        for c in caras:
            x1, y1, x2, y2 = [int(v) for v in c['caja']]
            if c['id_est']:
                inf = self.info[c['id_est']]
                reg = self.asistencia.get(c['id_est'])
                color = (0, 200, 0, 255) if reg else (255, 190, 0, 255)
                lineas = [f"ID: {c['id_est']}", f"{inf['apellidos']}, {inf['nombres']}",
                          f"Asistencia: {reg['fecha']} {reg['hora_ingreso']}" if reg else 'Verificando identidad...']
            else:
                color, lineas = (230, 40, 40, 255), ['Desconocido']
            d.rectangle([x1, y1, x2, y2], outline=color, width=3)
            alto = 20 * len(lineas) + 6
            ancho = max(d.textlength(l, font=self.f_bold) for l in lineas) + 12
            ty = y2 + 4 if y2 + alto + 4 < lienzo.size[1] else max(0, y1 - alto - 4)
            tx = min(max(0, x1), max(0, W - ancho))
            d.rectangle([tx, ty, tx + ancho, ty + alto], fill=(0, 0, 0, 190))
            for k, l in enumerate(lineas):
                d.text((tx + 6, ty + 3 + 20 * k), l, font=self.f_bold if k < 2 else self.f_normal, fill=color)
        # barra superior de estado
        d.rectangle([0, 0, W, 48], fill=(31, 56, 100, 210))
        d.text((8, 4), f"{self.ahora().strftime('%d/%m/%Y %H:%M:%S')}   |   "
                       f"Presentes: {len(self.asistencia)}/{len(self.info)}", font=self.f_bold, fill=(255, 255, 255, 255))
        d.text((8, 26), self.ultimo_evento, font=self.f_normal, fill=(170, 255, 170, 255))
        return lienzo

