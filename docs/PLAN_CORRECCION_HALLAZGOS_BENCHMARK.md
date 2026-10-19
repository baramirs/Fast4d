# Plan: corregir los hallazgos del benchmark de errores

Fecha: 2026-10-18. Estado: **Paso 0, A1, B1, B2 (con pregunta) y boton Test implementados** (2026-10-18, validados en la nube). Falta: corrida oficial Windows, A2/B3 como gate aparte, fase C.
Origen: `docs/PROTOCOLO_BENCHMARK_ERRORES.md`, seccion 6c (hallazgos 1 a 5).

Principio: Fast4D no debe entregar un mapa que se ve normal cuando la calibracion fallo.
Si la calibracion no es confiable, la GUI se detiene o pide confirmacion explicita.
Cada correccion se acepta solo si el benchmark lo demuestra (criterio numerico abajo).

---

## Diagnostico (causa probable)

| # | Hallazgo | Causa probable |
|---|---|---|
| 1 | Q pixel atascado (+4.7 %) en cristal sin strain, sin aviso | `crystal.calibrate_pixel_size` (py4DSTEM) usa `curve_fit`, un ajuste **local** que parte del valor inicial. Sin dispersion de |g| el perfil es muy angosto y el ajuste casi no se mueve del valor inicial (0.021 -> 0.020938). El aviso solo mira la distancia al valor inicial (< 5 %), asi que no salta. |
| 2 | Valor inicial 10-25 % errado: el ajuste salta a +55-92 % en 1-3 de 5 semillas | Mismo ajuste local: cae en otro minimo (otra familia de anillos). Luego el indexado del marco usa un Q pixel malo y elige otros planos (-35 o -55 deg). |
| 3 | Elipse de 1 % sin corregir: falla siempre | La elipse ensancha y desdobla los anillos; el ajuste de Q pixel cae en un minimo falso. La correccion de elipse existe pero esta apagada por defecto y nada detecta que hace falta. |
| 4 | El aviso de Q pixel salta tambien cuando el ajuste es correcto | El criterio es "distancia al valor inicial", no "calidad del ajuste". |
| 5 | Strain solo con ROI (sin global) da AttributeError | El paso de base deja `strainmap_full` sin ajustar (sin `g1g2_map`); `compute_strain(use_roi=True)` solo revisa si es `None`. |

Paso 0 (antes de cambiar codigo): confirmar las causas 1-3 con un barrido del puntaje del
ajuste vs Q pixel (perfil de correlacion entre |g| medidos y anillos teoricos) en los casos
`03_unstrained`, `q_guess=0.022` y `05_ellipse_1pct`. Si la causa no es la esperada, se
replantea el plan antes de seguir.

---

## Fase A: arreglos de bajo riesgo

**A1. Hallazgo 5 (ROI sin global).**
- Cambio: en `compute_strain(use_roi=True)`, si `strainmap_full` no tiene `g1g2_map`, calcular
  primero el mapa global (igual que hoy cuando es `None`), con mensaje en el log.
- Test nuevo: calibracion -> `compute_strain(use_roi=True)` directo debe dar el mismo mapa que
  global + ROI (`test_with_roi_without_global_first.py`).
- Riesgo: bajo. No cambia ningun resultado existente.

**A2. Metrica de calidad del ajuste de Q pixel (base para B y C).**
- Nueva funcion `q_pixel_fit_quality(braggpeaks, crystal, px)`: fraccion de picos (sin el haz
  central) cuyo |g| cae dentro de una tolerancia de un anillo teorico, y residuo RMS radial.
  Solo lectura, no cambia el ajuste.
- Se guarda en `scan.params` y en el reporte junto a `q_px_fitted`.
- Test: en los casos sinteticos, calidad alta cuando el ajuste es correcto y baja en los 3 casos
  que fallan (`q_guess=0.022` semillas 0/1/4, `ellipse 1 %`, `unstrained`).

## Fase B: ajuste de Q pixel robusto (hallazgos 1, 2 y 4)

**B1. Busqueda global antes del ajuste local.**
- Antes de `calibrate_pixel_size`, barrer Q pixel en una ventana alrededor del valor inicial
  (propuesta: +-30 %, paso 0.2 %) con la metrica A2; tomar el mejor y recien ahi hacer el ajuste
  local de py4DSTEM partiendo de ese valor. Se mantiene el ajuste de py4DSTEM (no se reemplaza).
- La ventana y el paso van en `param_spec.py` (explicitos, visibles en la GUI y en el reporte).

**B2. Reemplazar el aviso por un criterio de calidad.**
- Si la calidad A2 queda bajo un umbral: **la calibracion se marca como fallida** y la GUI no
  sigue al strain sin confirmacion explicita del usuario (y queda anotado en el reporte).
- El aviso de distancia al valor inicial pasa a ser informativo (ya no "WARNING").

**B3. Bloquear el indexado del marco con un Q pixel no confiable.**
- Hoy, si el angulo de g1/g2 y el indexado difieren mucho, se avisa y se usa el indexado.
  Propuesta: si ademas la calidad A2 es baja, detener (el error de -35/-55 deg sale de ahi).

## Fase C: elipse (hallazgo 3)

**C1. Detectar que hace falta corregir la elipse.**
- Con los picos ya calibrados en origen: medir la anisotropia de |g| vs angulo azimutal para
  cada familia de anillos. Si supera un umbral (propuesta: 0.3 %), avisar "elipse detectada,
  activar correccion" y no seguir sin decision del usuario.
- Test: dispara con elipse 0.5 % y 1 %, no dispara con 0 %.

**C2. Ruta corregida en el benchmark.**
- Agregar al config un barrido `ellipse_corrected` (elipse 0.5 / 1 / 2 % con correccion activa)
  para medir cuanto error queda despues de corregir. Requiere que el benchmark pueda activar la
  correccion de elipse (hoy esta forzada a apagada en nivel 1).

---

## Criterios de aceptacion (medidos con `run_benchmark.bat`)

| Caso | Hoy | Meta |
|---|---|---|
| Q pixel, cristal sin strain (`e_max` 0 a 0.002) | +4.7 % sin aviso | error < 0.1 % o calibracion marcada como fallida |
| Valor inicial Q pixel 0.016 a 0.026 (ampliar barrido) | 1-3 de 5 semillas fallan | 5/5 semillas con error de Q < 0.1 % y angulo < 0.1 deg |
| Elipse 1 % sin corregir | mapa con RMSE 0.37 %, exy 0.52 %, sin bloqueo | se detecta y se detiene antes del strain |
| Elipse 1 % corregida | (no medido) | RMSE y exy iguales al caso base +-10 % |
| Aviso de Q pixel con ajuste correcto | salta (0.019 -> 0.020) | no salta |
| Todos los demas casos del benchmark | ver 6b | sin cambios (diferencia < 1e-10 en strain) |
| Suite de tests | 275 pasan | todos pasan + tests nuevos |

## Orden de trabajo y control

1. Paso 0 (diagnostico) -> confirmar causas.
2. A1 -> tests -> commit.
3. A2 -> tests -> commit (aun no cambia resultados).
4. B1 + B2 + B3 -> benchmark completo (nube) -> corrida oficial Windows -> commit.
5. C1 + C2 -> benchmark -> corrida oficial -> commit.
6. Actualizar `PROTOCOLO_BENCHMARK_ERRORES.md` (historial y hallazgos resueltos).

Cada paso en una rama aparte. No se toca codigo validado fuera de estas funciones.

## Decisiones pendientes (del desarrollador)

1. Ante calibracion no confiable: **detener** (recomendado) o solo avisar en rojo y seguir.
2. Ventana de busqueda de Q pixel: +-30 % (propuesto) u otra.
3. Umbral de anisotropia para pedir correccion de elipse: 0.3 % (propuesto) u otro.
4. Si B1 debe proponerse tambien a py4DSTEM (upstream) o quedar solo en Fast4D.

---

## Implementado (2026-10-18)

Decisiones del desarrollador: (1) avisar y preguntar si se quiere continuar; (2) ventana +-30 %
y usar el boton Test del dialogo Q-pixel para esto; (3) umbral de elipse sin base fisica aun
(ver abajo); (4) ampliar la explicacion (ver abajo).

Paso 0 (diagnostico) confirmo la causa: desde cada valor inicial de 0.014 a 0.026, el ajuste local
de py4DSTEM cae en distintos minimos (0.0200, 0.0209, 0.0325, 0.0343...). El "ring match"
(fraccion del histograma radial sobre anillos teoricos, tol 1 %) da 0.82-0.97 en el minimo
correcto y <= 0.73 en los falsos, asi que separa bien la cuenca correcta.

Cambios (respaldo de los originales: `Claude outputs/backup_20261018_antes_fix_qpixel/`):
- `engine.py`: `q_pixel_global_search` (grilla de ring match en guess*(1 +- span), ajustes locales
  de py4DSTEM desde los 3 mejores puntos y desde el valor inicial; dentro de la cuenca ganadora se
  conserva el ajuste desde el valor inicial, asi los casos buenos no cambian) y
  `q_pixel_search_figure`. `calibrate_q_pixel` la usa siempre (refit ON) y guarda
  `params.q_px_check` (ok / differs / ambiguous). El aviso de +-5 % paso a "info".
  Nuevos parametros: `q_search_span` (0.30) y `q_search_tol` (0.01), en la tabla Q-pixel.
  A1: `compute_strain(use_roi=True)` calcula el global si el StrainMap no esta ajustado.
- `driver.py`: `compute_scan`/`compute_all` aceptan `confirm(scan, mensaje) -> bool`; si la
  busqueda marca differs/ambiguous, pregunta antes del strain (No = se detiene ese archivo).
  Sin `confirm` (batch, tests, benchmark) solo queda el WARNING en el log.
- `qt_main.py`: pregunta Si/No en la GUI (`_confirm_from_worker`); dialogo Q-pixel con
  "Test mode" (Global search por defecto / Fine sweep 2N+1 anterior) y "search span";
  Global search muestra la curva y pone el mejor valor en la casilla del valor inicial
  (no cambia la calibracion aplicada hasta "Apply calibration").
- Tests nuevos: `tests/test_q_pixel_global_search.py` (8) y `tests/test_qpixel_dialog_global_test.py` (1).
  Suite: 284 pasan, 2 omitidos.

Resultado benchmark nivel 1 en la nube (`benchmark_results/cloud_20261018_221816_qfix.tgz`):
- 116 de 130 filas identicas a la corrida anterior (strain < 1e-10, Q pixel < 1e-6 %).
- Las 14 filas que cambian son exactamente las que fallaban, y ahora estan bien:
  e_max 0.001: Q +4.69 % -> < 0.02 %; q_guess 0.022/0.025: Q +55..+92 % -> < 0.08 %, angulo
  -55/-35 deg -> 0.03 deg, RMSE eyy 0.74 % -> 0.065 %; elipse 1 % sin corregir: Q +62 % -> -0.70 %
  (lo que queda es la elipse misma), exy espurio 0.52 % -> 0.044 %.
- Barrido extra `config_qfix_check.json` (valor inicial 0.016-0.026, e_max 0-0.002): todo correcto.
- Costo: la calibracion de Q pixel hace ~4-5 ajustes locales en vez de 1 (tests: 120 s -> ~190 s).
  En datos reales grandes medir el tiempo en la corrida oficial.

Pendiente: corrida oficial Windows (`run_benchmark.bat`) y comparar con la nube; luego commit
en una rama (p. ej. `fix/qpixel-global-search`).

## Sobre la decision 3 (umbral de elipse)

No hay que inventar el umbral: sale de la precision que se necesita y del ruido.
- Una elipse no corregida de amplitud e = a/b - 1 deforma todos los |g| de forma anisotropa.
  En un mapa con orientacion de red uniforme casi todo se cancela al restar la referencia
  (el benchmark lo muestra: 0.2-1 % de elipse no cambia el RMSE de strain una vez que Q pixel
  es correcto). Pero si la red gira dentro del mapa, o si se reportan valores absolutos
  (d-spacing, Q pixel), el error es del orden de e.
- Propuesta: umbral = el menor de (precision objetivo de strain del experimento) y
  (3 x la incertidumbre de la anisotropia medida). Ej.: si se quiere strain a 0.1 %, corregir
  toda elipse > 0.1 % que sea estadisticamente significativa.
- Para fijarlo con datos: agregar al benchmark un caso con rotacion de red dentro del mapa
  (no existe aun en el generador) y barrer la elipse; el umbral es donde el error supera la
  precision objetivo.

## Sobre la decision 4 (py4DSTEM upstream)

- Solo Fast4D (hecho): rapido, bajo tu control, no depende de nadie. Contra: si otro script o
  notebook usa `calibrate_pixel_size` directo, sigue con el problema.
- Proponerlo a py4DSTEM: abrir un issue con el caso minimo (cristal sin strain, valor inicial
  5 % alto -> ajuste atascado; valor inicial 10 % alto -> otra familia de anillos) y, si lo
  aceptan, un PR con la busqueda global como opcion de `calibrate_pixel_size`. Beneficia a
  todos los usuarios y puede ser un aporte citable. Contra: tiempo de revision y tu py4DSTEM
  es una version `dev` (990d563): habria que rebasar el cambio sobre su rama actual.
- Recomendacion: mantener el arreglo en Fast4D ahora y abrir el issue con los datos sinteticos
  ya exportados (`benchmark_results/synthetic_level1/03_unstrained`) como ejemplo reproducible.

## Ajuste posterior: discos detectados (ronda 2, H-R2-6) - 2026-10-18

Con los braggpeaks DETECTADOS de la ronda 2 (CL160 3.0 mrad, 1e5) la primera version del
arreglo seguia fallando: el ajuste local de py4DSTEM, partiendo del maximo de la grilla, se
alejaba de la coincidencia con los anillos (las posiciones detectadas tienen sesgo) y ganaba el
ajuste desde el valor inicial (+60 %, marked "ambiguous"). Cambio: los maximos de la grilla son
candidatos por si mismos, y un ajuste local solo reemplaza al ganador si conserva >= 95 % de su
ring match (`Q_SEARCH_KEEP_FRAC`). Resultado: Q pixel -1.3 %, marco correcto, exy 0.00.
Analisis completo: `PARA TEST\strain_check_ronda2\ANALISIS_H-R2-6.md`.
Test nuevo: `tests/test_q_pixel_search_r2_detected.py` (+ fixture `tests/fixtures/r2/`).
Precision del ajuste de py4DSTEM dentro de la cuenca correcta: ~0.1-0.15 % segun el punto de
partida (tolerancia de los tests: 0.2 %). El strain no depende de eso.
Benchmark nivel 1 despues del cambio: strain identico a la corrida anterior
(`benchmark_results/cloud_20261018_231921_qfix2.tgz`). Suite: 285 pasan, 2 omitidos.
