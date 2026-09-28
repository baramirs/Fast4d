# Fast4D: cambios (resumen simple)

Lista corta de lo que se agrego o corrigio, del mas viejo al mas nuevo. El detalle de cada
punto esta en el mensaje del commit (`git log`).

## Julio 2026: base y memoria

- GUI y pipeline de Fast4D sobre py4DSTEM 0.14.19 (fijado en requirements).
- Ventana "Create ADF/BF" con selector de DP y progreso en vivo.
- Memoria: memmap por defecto (`FAST4D_FORCE_MEMMAP`), liberar los datos pesados de los scans
  inactivos y al terminar cada archivo, limite de scans en RAM configurable, deteccion de
  Bragg por lotes guardada en disco (streaming), liberar el datacube despues de guardar los
  braggpeaks.
- Interfaz: paneles reorganizados y guardados, barra de Analisis, "Auto-save on Compute".
- Report: arbol de figuras, exportacion a PDF / DOCX / PPTX, menu Tools.
- Q pixel: calcula el origen automaticamente si falta antes del ajuste.
- Licencia GPLv3, SECURITY.md, README.

## Septiembre 2026: herramientas nuevas

- Plugin de drift (`drift_estimate.py`): corrimiento entre scans repetidos (ADF / BF / strain,
  correlacion de fase), CSV por archivo; las lineas y ROIs se ubican sobre el mismo punto
  fisico en cada archivo.
- Plugin de reproducibilidad: metricas con y sin drift, box / violin y PCA.
- Herramienta unica Line / ROI (`qt_measure.py`): lineas con ancho, ROIs rotadas, en nm si hay
  paso de scan.
- Calibracion R: paso de scan (pm/px) leido del EMD de Velox (`r_px_pm`); distancias de
  lineas y figuras en nm.
- Seleccion de archivos para el analisis de sesion (guardada en el JSON de la sesion) y
  correlacion de canales sobre los archivos elegidos.
- Cache del probe del vacio; DP medio de una ROI.
- Sistema del cristal del tensor (`crystal_frame.py`): exx a lo largo de x_hkl (p. ej.
  [-110]) y eyy de y_hkl ([001]); tarjeta del sistema y rigidez del stress rotada a esos ejes.
- Indexado externo del BVM (`bvm_indexing.py`, RANSAC + hkl del cristal teorico): orienta el
  sistema del tensor (coordinate_rotation) con todos los planos indexados; figuras del BVM
  con hkl y con g1, g2, x, y; aviso si g1, g2 no son un par primitivo de la red.

## Septiembre 2026: correcciones del strain

- El tensor guardado ahora es el rotado por coordinate_rotation, con exx / eyy bien nombrados
  (antes se guardaba el sin rotar, con exx y eyy intercambiados).
- Los pixeles donde falla el ajuste (vacio) quedan fuera usando la mascara de py4DSTEM (antes
  salian como ~100 %).
- Signos de las etiquetas de g1 / g2 resueltos con el par medido (evita x a 90 deg).
- Modo 'apply': usa el Q pixel ajustado, no la semilla.
- GUI sin congelarse entre archivos en un Compute de varios archivos.

## 28-09-2026: correcciones de la auditoria engine vs notebook

Validadas con datos sinteticos de verdad conocida (Si [110]) y con Scan01. Tests en `tests/`
con un generador sintetico chico (`tests/synthetic_si110.py`).

- G1: la referencia "con ROI" usa ROI AND mascara de ajuste, como py4DSTEM (antes entraban
  pixeles sin ajuste y corrian todo el mapa hasta -1 % o +100 %).
- G2: se quito la mascara de "vacio" por ADF, que anulaba cristal real en campos sin vacio.
- G3: lineas y ROIs de analisis promedian solo pixeles con ajuste; los invalidos quedan
  vacios en los CSV y se reporta n valido (antes contaban como 0 %).
- M1: `ellipse_center` [0, 0] o fuera del BVM = usar el origen calibrado.
- M2: control de calidad antes de aplicar una elipse (a/b, centro, escala); avisos si el Q
  ajustado se aleja de la semilla o si la ROI es chica.
- M3: el modo 'apply' aplica la elipse ajustada (nuevo parametro de salida `ellipse_p_fitted`).
- M4: una ROI de calibracion que no se puede aplicar da error (antes usaba todo el scan en
  silencio).
- M5: las figuras de strain eligen la escala por panel (antes salian en blanco con strain
  uniaxial).
- D1 (sin cambiar valores por defecto): medida aparte la deteccion con minRelativeIntensity
  0.001 / q_kmax 1.6 contra los valores por defecto 0.05 / 1.0.
