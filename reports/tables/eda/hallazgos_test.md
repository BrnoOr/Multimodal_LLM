# Hallazgos del EDA — M3DI `test` (10000 escenas)


## Integridad

- 10000 filas, 25 columnas; nulos/NaN: ninguno.
- image_id únicos: 10000/10000 ✓
- row_idx contiguo 0..N-1: True; en orden: True.
- número del archivo de imagen crece con row_idx: True (condición para la alineación imagen↔fila).
- captions únicos: 5385/10000 (53.8%); las repeticiones son esperables en texto generado por plantillas.
- imágenes accesibles (muestra de 200): 100.0% (vía image_path directo: 0.0%).

## Imagen vs texto

- object_shape: 0.00% de desacuerdo.
- object_xpos: 0.00% de desacuerdo.
- object_ypos: 0.00% de desacuerdo.
- object_zpos: 0.00% de desacuerdo.

## Latentes

- object_zpos es constante (0): se excluye de la evaluación.
- latentes en [0,1]: ['object_alpharot', 'object_betarot', 'object_gammarot', 'object_color', 'spotlight_pos', 'spotlight_color', 'background_color'].
- object_alpharot: rango [0.000, 1.000], KS D=0.007 vs uniforme.
- object_betarot: rango [0.000, 1.000], KS D=0.009 vs uniforme.
- object_gammarot: rango [0.000, 1.000], KS D=0.013 vs uniforme.
- object_color: rango [0.000, 1.000], KS D=0.011 vs uniforme.
- spotlight_pos: rango [0.000, 1.000], KS D=0.007 vs uniforme.
- spotlight_color: rango [0.000, 1.000], KS D=0.007 vs uniforme.
- background_color: rango [0.000, 1.000], KS D=0.013 vs uniforme.

## Discretos

- object_shape: K=7, frecuencias {0: 1402, 1: 1440, 2: 1449, 3: 1406, 4: 1427, 5: 1421, 6: 1455}, min/max = 0.964, χ² uniforme p=0.94.
- object_xpos: K=3, frecuencias {0: 3429, 1: 3206, 2: 3365}, min/max = 0.935, χ² uniforme p=0.0192.
- object_ypos: K=3, frecuencias {0: 3331, 1: 3392, 2: 3277}, min/max = 0.966, χ² uniforme p=0.37.
- x ⟂ y: χ²=5.9, gl=4, p=0.206; celda mín/máx = 1065/1193.

## Caption

- object_xpos: mapeo inducido {0: 'left', 1: 'center', 2: 'right'}; inyectivo=True; P(w_k|k) media=1.000, P(w_j|k≠j) media=0.000.
- object_ypos: mapeo inducido {0: 'top', 1: 'mid', 2: 'bottom'}; inyectivo=True; P(w_k|k) media=1.000, P(w_j|k≠j) media=0.000.
- object_shape: mapeo inducido {0: 'teapot', 1: 'hare', 2: 'dragon', 3: 'cow', 4: 'armadillo', 5: 'horse', 6: 'head'}; inyectivo=True; P(w_k|k) media=1.000, P(w_j|k≠j) media=0.000.

## Color

- 99 nombres distintos; por fuente: tab=8 nombres/3376 filas, xkcd=69 nombres/3333 filas, css/base=22 nombres/3291 filas; no resolubles por matplotlib: ninguno.
- 61 nombres contienen espacio o '/' (p.ej. ['xkcd:bright yellow green', 'xkcd:greenish turquoise', 'xkcd:vivid purple', 'xkcd:deep sky blue']): cuidado con regex ingenuas.
- nombre más frecuente: tab:cyan (5.79%); exactitud por azar en nombre exacto (Σp²) = 0.0231.
- object_color_index ↔ nombre biyectivo: True (máx nombres por índice=1, máx índices por nombre=1).
- captions con nombre de color entre comillas: 100.0%; coincide con text_object_color_name: 100.0%.
- S de los nombres: mediana 0.98 [min 0.46]; V: mediana 1.00 [min 0.63]; acromáticos (S<0.15): 0.
- |matiz(nombre) − object_color| en filas cromáticas (100.0%): mediana 0.0136, p95 0.0789, máx 0.1434; <0.05: 88.6% (1 unidad = 360°).
- regla H1 matiz: reproduce el nombre del caption en 27.1% de las filas.
- regla H2 Lab HSV(h,1,1): reproduce el nombre del caption en 26.8% de las filas.
- regla H3 Lab HSV(h,0.98,1.00): reproduce el nombre del caption en 27.0% de las filas.
- nombres con mayor dispersión de matiz latente (desv. circular): tab:cyan=0.052, tab:red=0.041, tab:olive=0.039.
- término básico (desde el nombre): green=20.1%, red=13.8%, cyan=13.0%, blue=12.2%, yellow=11.2%, magenta=9.7%, orange=7.5%, purple=4.9%, gray=4.2%, pink=3.4%; mayoritaria=20.1%, Σp²=0.124; términos nunca usados: ['black', 'white'].
- acuerdo término básico nombre↔latente: 79.0% (el resto son colores frontera entre dos términos).
- |d(objeto, fondo)| < 0.05 (≈18°): 9.8% de las filas; < 0.10: 20.2%. Bajo independencia uniforme se esperaría 10% y 20%.

## Píxel

- 2000 imágenes analizadas; tamaños {(224, 224): 2000}; segmentación válida 100.0% ({'matiz+ΔE': 1907, 'ΔE': 93}).

## Posición

- object_xpos: centroide medio por código {0: 0.297, 1: 0.496, 2: 0.68}; código 0 = izquierda y código 2 = derecha en la imagen; ρ=0.939; exactitud de un clasificador por centroide=95.9%.
- object_xpos: palabra del caption para el código de izquierda = «left», para el de derecha = «right» → CONSISTENTE con la geometría.
- object_ypos: centroide medio por código {0: 0.39, 1: 0.51, 2: 0.677}; código 0 = arriba y código 2 = abajo en la imagen; ρ=0.850; exactitud de un clasificador por centroide=85.6%.
- object_ypos: palabra del caption para el código de arriba = «top», para el de abajo = «bottom» → CONSISTENTE con la geometría.

## Píxel: color y tamaño

- |d(matiz píxel, object_color)|: mediana 0.0309, p95 0.0562 → object_color no coincide limpiamente con el matiz renderizado (revisar iluminación/segmentación).
- correlación sen(2π·d(foco,objeto)) vs error de matiz: r=-0.233 (p=4.1e-26).
- |d(matiz del marco, background_color)|: mediana 0.3676.
- área mediana por forma (% imagen): {0: 3.27, 1: 3.17, 2: 1.15, 3: 2.52, 4: 3.13, 5: 2.0, 6: 3.57}.
- área mediana por object_xpos: {0: 2.99, 1: 2.23, 2: 2.87} (Kruskal–Wallis p=1.4e-11) → el tamaño aparente depende de la posición (perspectiva).
- área mediana por object_ypos: {0: 1.98, 1: 2.72, 2: 4.51} (Kruskal–Wallis p=2e-37) → el tamaño aparente depende de la posición (perspectiva).
- contraste objeto–fondo ΔE76: mediana 52.5; ΔE<20 (bajo contraste): 7.0% de la muestra.

## Líneas base

- por atributo descrito: forma: K=7, H/Hmax=1.000, Σp²=0.143, mayoritaria=0.145; xpos: K=3, H/Hmax=1.000, Σp²=0.334, mayoritaria=0.343; ypos: K=3, H/Hmax=1.000, Σp²=0.333, mayoritaria=0.339; color_basico: K=10, H/Hmax=0.946, Σp²=0.124, mayoritaria=0.201
- exactitud macro por azar (media Σp² sobre forma, x, y, color básico) = 0.234; con clase mayoritaria = 0.257.

## Dependencias

- pares con dependencia significativa y no trivial (p<1e-3, I−sesgo>0.01 bits): xpos–color_nombre: I=1.365 bits (sesgo≈0.014); xpos–color_basico: I=1.085 bits (sesgo≈0.001)

## Caption

- 5 plantillas; longitud 11–21 palabras (media 14.6); captions que mencionan posición: 100.0%.
