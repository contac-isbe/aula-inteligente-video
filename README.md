# Aula inteligente — Video

Proyecto de Percepción Computacional para UPAO. Primera etapa: ejecución **local en Windows**, independiente de Google Colab. La aplicación web es una etapa posterior.

## Qué hace esta versión

- Abre el video de una cámara integrada o webcam USB seleccionada por índice.
- Detecta rostros con MTCNN y compara embeddings FaceNet con estudiantes registrados.
- Muestra ID, apellidos y nombres sobre el video.
- Confirma una identidad tras tres frames consecutivos y registra asistencia una vez por día, con hora de Lima.
- Permite registrar un estudiante desde el video con `R`, ingresando sus datos manualmente.
- Conserva fotografía, ficha y base facial como apoyo al registro.

Esta etapa no incluye YOLO, ByteTrack, conteo de personas, grabación de video, métricas ni reporte de ausentes. Los rostros desconocidos se muestran como «Desconocido»: el usuario decide cuándo registrarlos para evitar interrupciones por visitantes o detecciones ambiguas.

## Instalar en Windows

1. Instalar **Python 3.11 de 64 bits** desde https://www.python.org/downloads/ (incluyendo el lanzador `py`) y Git desde https://git-scm.com/downloads/win.
2. Descargar o clonar este repositorio y abrir su carpeta.
3. Ejecutar `instalar.cmd`. Se crea un entorno `.venv` y se instalan las dependencias.
4. Ejecutar `iniciar.cmd`. La primera ejecución necesita Internet para descargar los pesos de FaceNet. Luego el reconocimiento se ejecuta en la computadora.

La configuración fija versiones compatibles para CPU; no requiere tarjeta gráfica NVIDIA.

```powershell
git clone https://github.com/contac-isbe/aula-inteligente-video.git
cd aula-inteligente-video
.\instalar.cmd
.\iniciar.cmd
```

## Cámara integrada o USB

Desde una terminal en la carpeta del proyecto:

```powershell
.\iniciar.cmd --camara 0
.\iniciar.cmd --camara 1
```

Los índices dependen del equipo: 0 no siempre es la integrada y 1 no siempre es USB. Si no abre, prueba otro índice y cierra Teams, Zoom u otras aplicaciones que usen la cámara. Comprueba el acceso de aplicaciones de escritorio a la cámara en la configuración de privacidad de Windows.

Para comprobar únicamente el video, sin descargar ni cargar modelos:

```powershell
.\iniciar.cmd --solo-video --camara 0
```

`R`: registrar la persona visible (debe aparecer exactamente un rostro). `Q`, Escape o cerrar ventana: finalizar y liberar la cámara. El video se pausa durante los formularios de registro.

## Datos locales

`data/` contiene estudiantes, rostros, fichas, embeddings y CSV diarios. Está excluida de Git. Cada integrante tiene su propia base; clonar el código no copia los registros de otra computadora. Registra participantes que autoricen el uso de su rostro. No subas datos reales al repositorio público.

El umbral 0.60 proviene del prototipo y todavía necesita calibración. La similitud es una puntuación, no un porcentaje de certeza. Tres frames reducen decisiones aisladas pero no garantizan identificación correcta. Esta versión no incorpora detección de suplantación ni evaluación experimental.

## Organización y siguiente etapa

- `aula/core.py`: lógica migrada del avance de Colab; reconocimiento, registro y asistencia.
- `aula/local.py`: video y formularios locales.
- `requirements.txt`: dependencias compatibles con Python 3.11.
- `tests/`: comprobaciones de registro y asistencia sin cámara ni descarga de modelos.

La futura web reutilizará el núcleo y sustituirá la ventana local por una interfaz en el navegador. Todavía no hay servidor web ni despliegue.

## Validación

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Estas pruebas no miden la precisión facial. Antes de presentar: comprobar ambas cámaras disponibles, registrar un participante autorizado, reconocerlo durante el video, verificar el CSV y comprobar que volver a entrar no duplica la asistencia.

Validación inicial (26/09/2026): Python 3.11.9 en Windows, cuatro pruebas aprobadas, dependencias sin conflictos y carga de FaceNet en CPU con detección sobre imagen sintética aprobada. Los índices de cámara 0 y 1 no entregaron video en el equipo de preparación; la prueba real de cámara y reconocimiento de participantes queda pendiente.
