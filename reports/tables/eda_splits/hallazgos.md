# Hallazgos del EDA de splits — M3DI base (train / val / test)


## Integridad

- train/val/test = 125,000 / 10,000 / 10,000 filas; ids únicos, row_idx contiguo y sin nulos/NaN: ✓.
- object_zpos es constante en los tres splits (no aporta información; se excluye).
- imágenes accesibles en disco: no (raíz C:\Users\bruno\Documents\MDS\Redes_Neuronales_Teoria_Informacion\Proyecto T7\data\raw\m3di); las secciones que las usan se omiten.

## Fuga

- sin ids, rutas ni escenas repetidas entre splits ni dentro de ellos.
- distancia mediana al vecino de train: test 0.2338, val 0.2344, train→train 0.2334 (KS test vs train: D=0.011; val vs train: D=0.012). No hay escenas casi duplicadas: test y val son tan lejanos a train como train de sí mismo.

## Captions

- captions únicos: train 10,936, val 5,372, test 5,385; el 99.4% de las filas de test tiene un caption que aparece literal en train.
- banco de VL-JEPA (20,000 filas de train → 7,492 captions únicos) contiene el caption correcto en el 85.5% de test: techo de la recuperación exacta con el objetivo original. Con el canónico la cobertura es 100.0%.

## Imagen vs texto

- forma, x e y coinciden en el 100 % de las filas de los tres splits: el techo por atributo es 1.

## Discretos

- los seis atributos discretos tienen la misma distribución en los tres splits: V de Cramér máx. 0.0092 (celda (x,y)), JSD máx. 2.86e-04 bits.
- en train: forma H/Hmax=1.000, x H/Hmax=1.000, y H/Hmax=1.000, celda (x,y) H/Hmax=1.000, color básico H/Hmax=0.950, plantilla H/Hmax=1.000. Forma, posición y plantilla son ~uniformes; el color básico no (las 8 categorías cubren arcos de matiz distintos).

## Color

- la frecuencia de cada categoría básica coincide con el ancho de su arco de matiz (desvío máx. 0.0019); va de 6.9% (orange) a 25.0% (green).

## Composicional

- las 63 combinaciones están presentes en los tres splits (mín. train 1,912, val 134, test 126); con el holdout, train pierde 27,643 filas (22.1%, quedan 97,357) y test evalúa 2,219 imágenes de combinaciones no vistas.
- cada forma pierde 2 celdas y cada celda se retira para top-left: 2, top-center: 1, top-right: 2, mid-left: 1, mid-center: 2, mid-right: 1, bottom-left: 2, bottom-center: 1, bottom-right: 2 formas: ninguna forma ni celda desaparece de train.

## Continuos

- los 7 latentes continuos son ~uniformes en [0,1] y homogéneos entre splits (KS train-val/test D máx. 0.0148).
- objeto y fondo con matiz a menos de 18° (camuflaje): train 10.1%, val 10.2%, test 9.8% — lo esperado bajo independencia (10 %).

## Color

- vocabulario de nombres: train 100, val 99, test 99; nombres de val/test ausentes en train: 0.
- H(nombre)=5.96 bits vs H(categoría)=2.85 bits; I(nombre; categoría)=2.50 bits → conociendo la categoría quedan 3.46 bits de incertidumbre sobre el nombre. 78% de los nombres caen siempre en una sola categoría.

## Captions

- longitud media: train 12.93, val 12.91, test 12.91 palabras (rango 10–19); vocabulario: train 114, val 114, test 114; palabras de test ausentes en train: 0.
- objetivo canónico: 231 textos distintos en train (no 7·9·8 = 504: ver §9.1), con 216–1016 filas cada uno; el original: 10,936 textos, 6% de ellos vistos una sola vez.

## Dependencias

- pares con dependencia significativa (G-test, Bonferroni α=0,001): train: x–color básico (NMI 0.706); val: x–color básico (NMI 0.707); test: x–color básico (NMI 0.708). Todos los demás pares tienen NMI < 0.0024 (independientes).
- el matiz del objeto está acotado por x: left [0.000, 0.333], center [0.333, 0.667], right [0.667, 1.000]. Solo 11 de 24 pares (x, color básico) existen → 231 captions canónicos posibles (no 504), lo que coincide con los 231 observados en train.
- atajo: adivinar x a partir del color acierta 80.9% en test (vs 33,3 % sin información) y el color a partir de x 42.5% (vs 24.7% con la clase mayoritaria).

## Azar

- exactitud de azar en test (mayoritario / marginal de train): color 0.247 / 0.152; color ±1 0.430 / 0.382; forma 0.145 / 0.143; nombre exacto 0.058 / 0.023; posición (x,y) 0.116 / 0.111; x 0.343 / 0.333; y 0.333 / 0.333.
