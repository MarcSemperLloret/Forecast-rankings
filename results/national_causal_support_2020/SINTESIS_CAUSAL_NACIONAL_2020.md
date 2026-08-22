# Síntesis causal nacional 2020

## Resultado principal

La intervención causal se generaliza fuera de AVAMET. En MIDAS-Reino Unido e
INMET-Brasil, los tres modelos primarios tienen un óptimo de suavizado positivo
contra ERA5 y nulo contra estaciones, tanto en MAE como en RMSE
(12/12 contrastes modelo–red–métrica pasan).

El suavizado óptimo contra ERA5 se sitúa entre 12,5 y 25 km. La reducción de
RMSE frente a ERA5 varía entre 0,006 y 0,121 °C y todos los intervalos
bootstrap de siete días para esos óptimos excluyen cero. En los seis contrastes
modelo–red, la reducción del componente centrado del MSE supera a la del sesgo².

## Escala de la referencia

La escalera prospectiva 0/50/100/200 km pasa en INMET (37
centros) y MIDAS (71 centros),
en MAE y RMSE, con media uniforme, ponderación gaussiana y thinning fijo a tres
estaciones. Todos los modelos presentan correlación de Spearman positiva; bajo
thinning, la correlación media de RMSE es 1,00 en ambas redes.

La escalera exacta prerregistrada 0/25/50/100 km pasa en MIDAS con
15 centros. INMET conserva solo
5 centros a 25 km y falla el gate de
geometría; no se interpreta como refutación científica.

## Afirmación permitida

> La dependencia causal de la evaluación respecto al soporte espacial de la
> referencia se generaliza desde AVAMET a dos redes nacionales, dos continentes,
> tres modelos y dos métricas.

## Límites

MIDAS e INMET no se solapan con Weather5K/ISD en el estrato seleccionado, pero
su ausencia de todas las rutas de asimilación de ERA5 no está documentada. Los
modelos comparten entrenamiento, inicializaciones o linajes y no constituyen
siete réplicas independientes. Sigue prohibida la expresión «confirmación
global completamente independiente de ERA5».

## Pie de figura

**Figura. El soporte espacial de la referencia controla la estructura que
recompensa la verificación.** **a,b**, cambio de RMSE al suavizar
geodésicamente el mismo pronóstico, relativo a σ=0, en INMET-Brasil y
MIDAS-Reino Unido. Las barras son IC bootstrap del 95% con bloques temporales
circulares de siete días. **c,d**, suavizado óptimo frente a referencias
observacionales construidas con soporte creciente. Las líneas continuas usan
media uniforme; las discontinuas mantienen tres estaciones por centro en 60
selecciones y dejan crecer únicamente el área cubierta. Los tres modelos se
muestrean en los mismos centros y tiempos en cada escalón. Los marcadores de
**c,d** llevan un desplazamiento horizontal mínimo para hacer visibles los
óptimos coincidentes; los radios analizados siguen siendo 0, 50, 100 y 200 km.
