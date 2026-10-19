# Protocolo: benchmark de errores de Fast4D contra valor real (sintetico)

Estado: **nivel 1 implementado y validado** (2026-10-18). Corrida oficial en Windows hecha y comparada con la nube.
Objetivo: medir, con datos sinteticos de strain conocido, el error que entrega el
pipeline de Fast4D (el mismo que usa la GUI) y poder repetir el ejercicio igual en el futuro.

---

## 1. Regla principal: siempre el py4DSTEM de la GUI

- Entorno de la GUI: conda `py4dstem-01419`
  - `C:\Users\jtapiaca.ASURITE\.conda\envs\py4dstem-01419\python.exe` (el mismo de `run_gui.bat`)
  - py4DSTEM **0.14.19** (esta version NO existe en PyPI ni como tag en GitHub; solo existe en ese entorno).
- Ningun resultado vale si se obtuvo con otra version. El script de benchmark debe:
  1. Verificar `py4DSTEM.__version__ == "0.14.19"`; si no, detenerse con error.
  2. Calcular un hash (SHA-256) de todos los `.py` del paquete py4DSTEM instalado y compararlo
     con el hash de referencia guardado en el repo. Si no coincide, detenerse con error.
  3. Guardar en cada resultado: version y hash de py4DSTEM, `pip freeze`, commit de Fast4D
     (y si hay cambios sin commit), sistema operativo, semilla(s).
- Llamar a las mismas funciones del engine que usa la GUI
  (`engine.run_calibration_sequence`, `engine.compute_strain`, ...), nunca una reimplementacion.

## 1b. Implementacion (archivos)

- `tools/benchmark/benchmark_errors.py`: script. Sin valores por defecto: todo sale del config.
  Aborta si la version o el hash de py4DSTEM no coinciden. Usa `engine.run_calibration_sequence`
  y `engine.compute_strain` en el mismo orden que `driver.py` (global, luego con ROI).
- `tools/benchmark/config_level1.json`: base + barridos (un factor a la vez, 5 semillas).
  Contiene la version y el hash de referencia de py4DSTEM.
- `tools/benchmark/plot_results.py`: figura `figures/errors_vs_factors.png/.pdf` de una corrida
  (se genera sola al final de cada benchmark; para una corrida antigua:
  `python tools/benchmark/plot_results.py benchmark_results/<carpeta>`).
- `tools/benchmark/export_synthetic.py`: exporta los datos sinteticos con su valor real a
  `benchmark_results/synthetic_level1/` (braggpeaks.h5, truth.npz, fast4d_measured.npz, case.json,
  maps.png). Verifica que el archivo cargado desde disco de el mismo error que en memoria.
  Uso: `python tools/benchmark/export_synthetic.py tools/benchmark/config_level1.json <carpeta_nueva>`.
- `run_benchmark.bat`: corrida oficial con el python del entorno `py4dstem-01419`.
  Uso: doble clic (config por defecto) o `run_benchmark.bat ruta\config.json`.
- Salida: `benchmark_results/<fecha>_<commit>_<SO>/` con `summary.csv`, `summary.json`,
  `per_seed.csv`, `env.json` (versiones, hash py4DSTEM, commit y hash del codigo Fast4D),
  `config.json` y `run.log` (warnings y mensajes del engine).
- Hash de referencia py4DSTEM 0.14.19: `de2defff32f2448a51724d5b26019633b3f7ad46eaa3eae44f3d8710d5f4b499`
  (receta: sha256 de cada `.py` del paquete, rutas ordenadas, y sha256 de esa lista).
  py4DSTEM proviene de git `dev` commit 990d563d91315db59044d4ee0ac895b95e5a5c3a.

Entorno nube (reconstruccion, 2026-10-18): Python 3.11.14, mismas versiones de los 191 paquetes
del entorno (lista de dist-info), py4DSTEM copiado byte a byte (hash identico).
Diferencias conocidas con Windows: sin `cupy` (no hay GPU en la nube), y BLAS distinto
(Windows: numpy de conda con MKL / openblas). Por eso la corrida oficial es en Windows.
Nivel 1 no usa deteccion de discos, asi que la GPU no influye; en nivel 2 si puede influir.

## 2. Flujo hibrido

| Paso | Donde | Para que |
|---|---|---|
| Desarrollo del script y barridos grandes | Nube (Cowork) con copia exacta de py4DSTEM 0.14.19 + `pip freeze` del entorno | Rapido, no ocupa el PC |
| Corrida oficial (la que se reporta) | Windows, entorno `py4dstem-01419`, via `run_benchmark.bat` | Garantia total: mismo interprete que la GUI |
| Comparacion nube vs Windows | Automatica | Si difieren mas de la tolerancia (propuesta: 1e-10 en strain), se marca y manda la de Windows |

Copia a la nube (una sola vez, repetir si cambia el entorno):
- Carpeta `Lib\site-packages\py4DSTEM` del entorno + `pip freeze` + hash de referencia.
- Reconstruir el entorno en la nube con esas versiones exactas (Python de la misma version menor).
- Validar: el caso base (seccion 6) debe dar el mismo resultado en nube y Windows.

## 3. Datos sinteticos

Nivel 1 (implementar primero): generador `tests/synthetic_si110.py`.
- Si [110], a = 5.431 A, picos de Bragg inyectados directamente en un `BraggVectors`.
- Strain fisico solo en e_yy (rampa lineal), origen del haz en un plano conocido, elipse opcional,
  jitter gaussiano en la posicion de picos, banda de vacio opcional.
- Ojo: **se salta la deteccion de discos**. Mide error de calibracion, ajuste de red y calculo
  de strain, NO el error de deteccion.

Nivel 2 (despues): patrones de difraccion simulados (discos con forma de sonda, ruido Poisson,
fondo), p. ej. con abTEM o el simulador de py4DSTEM, pasando por la deteccion de discos de Fast4D.
En datos reales el error de deteccion suele ser el dominante.

## 4. Factores a barrer

Prioridad nivel 1: factores 1, 3 y 4.

1. Ruido en posicion de picos: 0 a 0.2 px.
2. Magnitud del strain: 0.1 % a 10 % (efecto no lineal de e/(1+e)).
3. Errores de calibracion: centro, elipse no corregida, Q pixel inicial, angulo de rotacion
   de coordenadas (p. ej. cuanto strain espurio genera 1 grado de error).
4. Referencia: tamano del ROI, fraccion de vacio dentro del ROI, referencia con strain propio.
5. Tamano del scan y numero de reflexiones (g_max).

Diseno: factorial fraccionado (skill `experimental-design`) para ver que factores dominan y
sus interacciones; varias semillas por punto.

## 5. Metricas y salidas

Por cada condicion, contra `expected_eyy(truth, ref_mask)`:
- sesgo (media), RMSE, max |error|, percentil 95, en % de strain
- RMS de e_xx y e_xy espurios (deberian ser 0)
- Q pixel ajustado vs real
- intervalos de confianza del sesgo entre semillas (skill `statistical-analysis`)
- presupuesto de incertidumbre tipo GUM / Monte Carlo (skill `uncertainty-and-units`)
- figuras error vs factor (skill `scientific-visualization`)

Formato de salida (para gastar pocos tokens):
- Consola: solo una tabla resumen corta. Barras de progreso (tqdm) desactivadas.
- `results/<fecha>_<commit>/summary.json` + `summary.csv` + figuras + `env.json` (metadatos).
- Logs detallados a archivo; leerlos solo si algo falla.

## 6. Caso base ya verificado (2026-10-18, referencia)

Corrido en la nube con py4DSTEM 0.14.18 (NO es la version oficial; repetir con 0.14.19).
Scan 24x24, sin vacio, strain sin ROI, referencia = mediana de todo el cristal.

| e_max | jitter (px) | RMSE eyy | max |error| | RMS exx / exy |
|---|---|---|---|---|
| 0 | 0 / 0.03 | 0 / 0.067 % | 0 / 0.20 % | 0 / ~0.05 % |
| +0.02 | 0 / 0.03 | 0 / 0.067 % | 0 / 0.20 % | 0 / ~0.05 % |
| -0.02 | 0 / 0.03 | 0 / 0.067 % | 0 / 0.20 % | 0 / ~0.05 % |
| +0.10 | 0 / 0.03 | 0 / 0.098 % | 0 / 0.27 % | 0 / ~0.06 % |

Q pixel: ajustado 0.020001 1/A vs real 0.020 (valor inicial 0.021).
Suite de tests: 275 pasan, 2 omitidos (faltan archivos demo: Bragg peaks demo y H5 de simulacion).

## 6b. Resultados nivel 1 en la nube (2026-10-18, py4DSTEM 0.14.19 exacto, Linux, 5 semillas)

Base: n=24, e_max=0.02, jitter=0.03 px, frame_mode=crystal (default GUI), sin ROI.
Resultado completo: `benchmark_results/20261018_174235_0e8b993_Linux/` (en la nube; copiar si se necesita).

| Factor | Efecto principal |
|---|---|
| Ruido de picos | RMSE eyy crece lineal: ~2.15 % por px de jitter (0.024 % a 0.01 px; 0.43 % a 0.2 px). Sesgo ~0. |
| Magnitud strain | Sin sesgo hasta 2 %; a 10 % sesgo +0.027 % y RMSE 0.10 %. |
| Error de rotacion (modo manual) | exy espurio: 0.043 % (0 deg) a 0.106 % (5 deg); eyy casi no cambia. Modo crystal encuentra el angulo con error ~0.03 deg. |
| Error de centro inicial | Sin efecto hasta 10 px. |
| ROI con 40 % vacio | Sin offset (el fix G1 funciona); sesgo 0.013 %. |

## 6c. Hallazgos (revisar en el codigo, no corregidos)

1. **Calibracion de Q pixel atascada sin aviso en cristal sin strain.** Con e_max <= 0.002
   (incluye 0, el caso de una referencia sin deformar) el ajuste da 0.020938 con valor real 0.020
   (+4.7 %), y como esta a menos de 5 % del valor inicial 0.021, el engine no avisa.
   El strain no cambia (es relativo), pero cualquier distancia absoluta (1/A, d-spacing) queda mal.
2. **Valor inicial de Q pixel a 10-25 % del real: falla en 1 a 3 de 5 semillas.** El ajuste salta a
   0.032-0.038 (+55 a +92 %), el indexado del marco cristalino se equivoca (-35 o -55 deg) y aparece
   exy espurio de ~0.5 % y RMSE eyy 0.4-0.7 %. El engine avisa (warning), pero sigue y entrega mapas.
3. **Elipse no corregida de 1 %: falla siempre** (5/5): Q pixel +62 %, marco -35 deg, exy espurio 0.52 %.
   Con 0.2-0.5 % no hay efecto visible en strain (Q pixel -0.2 a -0.3 %).
4. **El aviso de Q pixel (limite +-5 % del valor inicial) tambien salta cuando el ajuste es correcto**
   (valor inicial 0.019, ajuste correcto 0.020 a +5.3 %). Avisa por distancia al valor inicial,
   no por calidad del ajuste: riesgo de ignorar avisos.
5. **Strain con ROI sin haber calculado antes el global falla** (AttributeError: StrainMap sin
   g1g2_map) si en la GUI se desmarca "without ROI" y se deja "with ROI". El paso de base deja un
   `strainmap_full` sin ajustar y `compute_strain(use_roi=True)` no lo detecta.

## 6d. Corrida oficial Windows vs nube (2026-10-18)

Oficial: `benchmark_results/20261018_182814_0e8b993_Windows/` (Windows 10.0.26300, Python 3.11.14,
py4DSTEM 0.14.19 hash de2defff32f2, codigo Fast4D d035b7c3688f, 1524 s).
Nube: `benchmark_results/cloud_20261018_174235.tgz` (mismo hash py4DSTEM y mismo hash de codigo).

Comparacion de las 130 filas (26 casos x 5 semillas):
- Todas las metricas de strain (sesgo, RMSE, p95, max, exx, exy), angulo y fallas de ajuste:
  diferencia maxima 1.7e-13 (% strain). Identicas a precision de maquina.
- Q pixel ajustado: diferencias de hasta 0.022 puntos porcentuales (p. ej. 0.0268 % vs 0.0057 %)
  en 73 filas. El optimizador cae en minimos ligeramente distintos segun BLAS/SO, pero el strain
  no cambia porque es relativo. Los casos de falla grande (hallazgos 2 y 3) coinciden en ambos.
- Conclusion: la nube reproduce el strain de la GUI. Tolerancias: strain 1e-10; Q pixel 0.05 pp.
  Los resultados de la seccion 6b valen tambien para Windows.

## 7. Como repetir el ejercicio (checklist)

1. Confirmar que el entorno `py4dstem-01419` no cambio (version + hash). Si cambio, actualizar
   el hash de referencia y la copia en la nube, y anotarlo aqui.
2. Nube: correr el barrido con la copia exacta; revisar `summary.csv`.
3. Windows: doble clic en `run_benchmark.bat` (corrida oficial).
4. Comparar nube vs Windows; si difieren mas de la tolerancia, investigar antes de reportar.
5. Anotar en la seccion 8 la fecha, commit, version/hash y conclusion.

Herramientas usadas en Cowork: plugin `scientific-pipeline-skills` (experimental-design,
uncertainty-and-units, statistical-analysis, scientific-visualization) y skill `scientific-coding`.
Idea pendiente: empaquetar este protocolo como skill "fast4d-error-benchmark".

## 8. Historial de corridas

| Fecha | Commit Fast4D | py4DSTEM (version / hash) | Donde | Conclusion |
|---|---|---|---|---|
| 2026-10-18 | 0e8b993 + cambios sin commit | 0.14.18 / - | Nube | Caso base sin sesgo; error ~ ruido de picos. No oficial. |
| 2026-10-18 | 0e8b993 + cambios sin commit (code d035b7c3688f) | 0.14.19 / de2defff32f2 | Nube (Linux) | Nivel 1 completo. Hallazgos 6c. |
| 2026-10-18 | 0e8b993 + cambios sin commit (code d035b7c3688f) | 0.14.19 / de2defff32f2 | **Windows (oficial)** | Igual a la nube en strain (max dif 1.7e-13). Q pixel difiere hasta 0.022 pp. |
| 2026-10-18 | 0e8b993 + fix Q-pixel (code 61d180f4643d) | 0.14.19 / de2defff32f2 | Nube (Linux) | Fix busqueda global Q pixel: hallazgos 1-3 y 5 resueltos; 116/130 filas identicas, 14 corregidas. Ver PLAN_CORRECCION. Falta oficial Windows. |
| 2026-10-19 | 0e8b993 + fix Q-pixel v2 (code 68aabeca71ee) | 0.14.19 / de2defff32f2 | **Windows (oficial)** | Nivel 1 con el arreglo: strain identico a la nube (1.7e-13); Q pixel < 0.16 % salvo elipse. |
| 2026-10-19 | idem | 0.14.19 / de2defff32f2 | Nube (Linux) | Nivel 2 (patrones 4D + deteccion): ver BENCHMARK_NIVEL2_RESULTADOS.md. Falta oficial Windows. |
