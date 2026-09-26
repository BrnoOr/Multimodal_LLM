# Proyecto: Descripción de escenas controladas con modelos multimodales: VLMs autorregresivos frente a VL-JEPA

## Descripción:

Un modelo de lenguaje multimodal (VLM) recibe una imagen y un prompt, y 
produce texto. La pregunta práctica es qué tan fiel es ese texto a lo que realmente hay en la 
imagen. En datos naturales esto es difícil de medir, porque no existe una descripción "correcta" 
única. Este proyecto usa en cambio un conjunto sintético donde cada imagen fue generada a 
partir de un conjunto pequeño de atributos conocidos (forma, color, posición y ángulo del objeto, 
color del foco y del fondo), y donde el texto de referencia describe explícitamente esos atributos. 
Eso permite comparar la descripción generada contra la verdadera atributo por atributo, y no solo 
con métricas de similitud textual.

## Etapas del proyecto:
1. Inferencia sin ajuste. Se evalúa cada modelo tal como viene preentrenado, con distintos prompts, midiendo cuánto se aproxima el texto generado al de referencia. Esto establece el punto de partida y expone el desajuste entre el dominio de preentrenamiento (imágenes web) y el dominio sintético del conjunto. 
2. Ajuste fino y comparación. Se hace fine-tuning de los tres modelos sobre el mismo conjunto y con el mismo presupuesto de cómputo, y se vuelve a medir. La comparación entre las dos etapas es tan informativa como la comparación entre modelos: indica cuánto del error inicial era falta de conocimiento y cuánto era simple desalineamiento de formato. 

## Modelos a comparar

**LLaVA**: como línea base: encoder visual congelado, proyector lineal y decodificador de 
lenguaje. 
**Un VLM moderno superior a LLaVA (Qwen2.5-VL o InternVL 3)**: con conector 
más expresivo y resolución dinámica. Permite separar qué limitaciones vienen del 
paradigma autoregresivo y cuáles solo de una arquitectura antigua. 
**VL-JEPA**: en lugar de generar tokens, predice el embedding continuo del texto 
objetivo a partir de la imagen y de la consulta, e invoca un decodificador ligero solo cuando se necesita texto legible. Es una forma distinta de resolver la misma tarea y el punto de contraste central del proyecto. 

Todos los modelos se usan preentrenados y cuantizados (4-8-bit NF4 o AWQ), con adaptación mediante LoRA. No se entrena nada desde cero. Si alguno de los modelos resulta suficientemente bueno, todo el framework puede adaptarse a datos clínicos reales del Instituto de Neurocirugía. 

## Métricas

Además de las métricas estándar de generación (BLEU, ROUGE, CIDEr), se debe 
reportar exactitud por atributo: extraer de la descripción generada el valor de cada atributo y 
compararlo con el verdadero. Esta es la medida que realmente responde la pregunta del 
proyecto, ya que un texto puede parecerse mucho al de referencia y equivocarse justo en el 
color o la posición.

## Datos
Multimodal3DIdent, conjunto de imágenes renderizadas y descripciones textuales 
con atributos generativos conocidos y controlados. Se usa tanto el conjunto publicado como su generador, lo que permite construir variantes (nuevas combinaciones de atributos, conjuntos fuera de distribución, descripciones con más o menos detalle) para probar generalización.


