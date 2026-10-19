# Benchmark nivel 2: condicion optima de adquisicion y analisis + rotaciones (2026-10-19)

Datos sinteticos dibujados (patrones 4D Merlin, discos de borde suave, ruido Poisson) que pasan
por la deteccion real de Fast4D -> calibracion -> strain. Generador adaptado de
`strain_check_ronda2/B/ronda2_B.py`. py4DSTEM 0.14.19 (hash de2defff32f2), Fast4D 0e8b993 +
arreglo de Q pixel. Corridas en la nube (CPU, `detect_cuda` false): falta la oficial en Windows con GPU.

Herramientas: `tools/benchmark/level2/` (gen4d.py, run_level2.py, analyze_level2.py, configs),
`run_benchmark_level2.bat`. Resultados: `benchmark_results/cloud_L2_stage{A,B}_*.tgz`,
figuras en `benchmark_results/L2_figures/`.

Escenario: Si [110], scan 12 x 12 (columnas 0-5 referencia, 6-11 deformadas), detector 256 x 256,
strain fisico exx -0.3 %, eyy +0.6 %, exy +0.2 % (marco del cristal), 3 semillas por caso.
Criterio: **RMSE total** = sqrt(promedio de RMSE^2 de exx, eyy, exy) en el bloque deformado
contra el valor real (convencion py4DSTEM). Sonda: d = 1.22 lambda / alpha (200 kV).

## Etapa A: adquisicion x deteccion (factorial completo, 72 condiciones, 216 corridas)

Factores: CL (q_px 0.0224 = CL98, 0.0130 = CL160), alpha 1.5 / 2.1 / 3.0 mrad,
dosis 1e4 / 1e5 / 1e6 e por patron, deteccion poly o multicorr x umbrales
"default" (minAbs 3, minRel 0.05) o "sensible" (minAbs 0, minRel 0.001).

RMSE total (%), detección sensible, multicorr:

| CL | alpha (sonda) | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|
| 160 | 1.5 mrad (2.0 nm) | 0.204 | **0.054** | **0.018** |
| 160 | 2.1 mrad (1.5 nm) | 0.319 | 0.063 | 0.021 |
| 160 | 3.0 mrad (1.0 nm) | 0.345 | 0.073 | 0.030 |
| 98 | 1.5 mrad | 0.313 | 0.100 | 0.029 |
| 98 | 2.1 mrad | 0.300 | 0.093 | 0.031 |
| 98 | 3.0 mrad | 0.408 | 0.101 | 0.039 |

Efectos (de mayor a menor):
1. **Dosis**: el RMSE baja ~3x por cada factor 10 (1/sqrt(dosis)). Es lo que mas pesa.
2. **Umbrales de deteccion "sensibles"** (minAbs 0, minRel 0.001): siempre mejores. Con los
   umbrales por defecto de Fast4D se pierde {004} y, a 1e4, la mitad de las condiciones fallan
   (LinAlgError, 6 de 12). A 1e5 el RMSE baja 30-40 % con los umbrales sensibles.
3. **CL mas largo (CL160)**: RMSE 1.5-2x menor que CL98 (mas pixeles por 1/A).
4. **alpha 3.0 mrad es el peor**: error de posicion sistematico 0.37-0.48 px (discos cerca
   de tocarse: el diametro 2 alpha = 6 mrad frente a 8 mrad de separacion {111}), sesgo en exy
   (-0.03 a -0.08 %), Q pixel -1.3 a -4 % y fallas de calibracion a baja dosis (abajo).
   Entre 1.5 y 2.1 mrad la diferencia es chica (~15 %).
5. **multicorr vs poly**: multicorr algo mejor (2-10 %) y reduce el sesgo a 3.0 mrad.
   Nota: en Fast4D "com" se traduce a "multicorr" (pipeline.normalize_subpixel_keyword).

**Condicion recomendada** (simulacion): CL160, deteccion multicorr con minAbs 0 y minRel 0.001,
la mayor dosis que la muestra tolere, y alpha 1.5-2.1 mrad. Si la resolucion espacial importa
(nanohojas GAA), 2.1 mrad (sonda ~1.5 nm) cuesta solo ~15 % mas de RMSE que 1.5 mrad (~2.0 nm).
Evitar 3.0 mrad con Si [110] a 200 kV.

## Etapa B: rotaciones (CL98, 2.1 mrad, 1e5, deteccion sensible, poly; qr real 12 deg)

| prueba | efecto |
|---|---|
| rotacion de la red 0-90 deg | ninguno (RMSE 0.093-0.101 %; marco < 0.15 deg) |
| rotacion que cambia dentro del mapa (hasta 5 deg) | theta seguido con ~0.05 deg; strain igual |
| qr con error +-1 a +-5 deg, o +180 | tensor igual; el angulo del marco reportado se corre lo mismo que el error |
| flip equivocado | **exy cambia de signo (sesgo -0.40 %) y marco -66 deg, sin aviso** |
| marco manual con error | exy espurio ~0.016 % por grado (5 deg: 0.08 %) |

Convencion verificada: theta de py4DSTEM tiene el sentido opuesto a la rotacion antihoraria
de la red en q calibrado.

## Hallazgos nuevos (no corregidos)
- **H-L2-1. Falla silenciosa de Q pixel a baja dosis cerca del solapamiento**: CL98 3.0 mrad,
  1e4, deteccion sensible: Q +57 % y marco -35.6 deg en 4 de 6 corridas con estado "ok" de la
  busqueda global (no pregunta). El log si trae el aviso "g1/g2-label angle differs by -35 deg
  from the indexed planes". Propuesta: usar ese desacuerdo (g1g2_vs_indexing) como segundo
  disparador de la pregunta al usuario.
- **H-L2-2. El flip no es detectable desde los datos de difraccion**: `y_sign` del marco no es
  confiable (-1 aqui, +1 en T1 de la ronda 2). Hay que calibrar el flip con una referencia
  fisica (imagen de sombra / sonda desenfocada, o una muestra de orientacion conocida).
- **H-L2-3. La cache de la sonda de vacio deja abierto el .mib del vacio** (memmap) y guarda
  .npz en `~/.fast4d/probe_cache`. En la GUI: el archivo de vacio queda bloqueado mientras la
  app esta abierta. En el benchmark se limpia con `clear_vacuum_probe_cache(disk=True)`.
- En la ronda 2, "subpixel com peor" (H-R2-7) era multicorr a 3.0 mrad; a 1.5-2.1 mrad multicorr
  es el mejor metodo.

## Limitaciones
Simulacion cinematica (sin difraccion dinamica, espesor, inclinacion, MTF del detector ni
forma real de la sonda); un solo estado de strain; scan 12 x 12; 3 semillas; deteccion en CPU.
Las cifras absolutas no se trasladan directo a datos reales; el orden relativo de condiciones si
es la guia. Validar con datos reales (Scan01-04) antes de cambiar el protocolo de adquisicion.

## Siguiente
1. Corrida oficial en Windows con GPU (`detect_cuda` true) de las dos etapas.
2. Decidir H-L2-1 (segundo disparador de la pregunta) y H-L2-3.
3. Repetir la etapa A con un strain mas chico (0.1-0.3 %), mas realista para GAA.
