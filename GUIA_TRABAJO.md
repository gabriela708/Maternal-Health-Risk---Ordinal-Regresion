# Guía de trabajo — Entrega final (sección 3.4 del enunciado)

**Proyecto:** Clasificación del nivel de riesgo materno con regresión logística ordinal bayesiana
**Entrega:** lunes de la última semana de clases, en Teams. Todos los archivos se nombran `GrupoX` (si no, se pierde una unidad).
**Qué se entrega:** PDF en la plantilla LaTeX (máximo **15 páginas** incluyendo portada y bibliografía), el código Python y la base de datos. No se puede usar Excel. El informe debe incluir el dominio de donde provienen los datos.

Esta guía dice **qué** hacer y **en qué orden**. El **cómo** lo deciden y lo justifican ustedes.

---

## Paso 0 — Todos (primer día)

1. [ ] Leer completa la sección 3.4 y la sección 4 (calificación) de `docs/Enunciado_Proyecto.pdf`.
2. [ ] Leer `docs/correcciones.md` y repartir cualquier punto que no esté asignado.
3. [ ] Conectar el repositorio siguiendo el `README.md`.
4. [ ] Acordar en grupo y completar `src/config.py`: semilla, nombres de variables y codificación ordinal de RiskLevel.
5. [ ] Descargar la plantilla LaTeX oficial del enlace del enunciado y pasar su preámbulo a `informe/main.tex`.
6. [ ] Fijar fechas internas para cada hito (ver cronograma al final).

---

## Persona A — Sara · Problema, estado del arte y EDA

**Archivos:** `src/01_eda.py`, `informe/secciones/00_resumen.tex`, `01_introduccion.tex`, `02_estado_arte.tex`, `03_eda.tex`

### Parte 1 — Pensamiento crítico (correcciones de la segunda entrega)
1. [ ] Releer el comentario de la profesora en la segunda entrega y la lista de `docs/correcciones.md`.
2. [ ] Escribir qué se entiende por **riesgo gestacional** y cómo se clasifica en niveles, apoyándose en una guía clínica.
3. [ ] Para **cada variable** de la base, explicar con fuentes su relación clínica con el riesgo.
4. [ ] Reunir los artículos clave y extraer de cada uno: objetivo, datos, método, hallazgos y limitaciones.
5. [ ] Reescribir el **estado del arte** integrando esos hallazgos en tres bloques: qué se sabe, qué se discute y qué falta investigar. No listar los artículos uno por uno.
6. [ ] Revisar los puntos de consistencia de `docs/correcciones.md` en lo que toca a referencias.
7. [ ] Pasar todas las referencias a `informe/Bibliografia.bib`.
8. [ ] Releer todo el texto en voz alta: quitar repeticiones y reescribir con palabras propias.

### Parte 2 — EDA (`src/01_eda.py`)
9. [ ] Completar los TODO del script.
10. [ ] Anotar todo problema de calidad que aparezca y comunicárselo a la Persona C, porque afecta el preprocesamiento.
11. [ ] Elegir solo las gráficas pertinentes para el problema (mínimo cuatro) y redactar qué muestra cada una.

### Parte 3 — Informe
12. [ ] Introducción: contexto, problema, pregunta, justificación, objetivos y estructura del documento.
13. [ ] Sección de análisis exploratorio.
14. [ ] Resumen y abstract (se escriben al final, cuando estén los resultados).

### Parte 4 — Presentación (15 %)
15. [ ] Diapositivas de contexto, problema, estado del arte y EDA.

### Parte 5 — Revisión cruzada
16. [ ] Revisar el Pull Request de la Persona B.

---

## Persona B — Alejandra · Especificación del modelo bayesiano

**Archivos:** funciones de la sección "PERSONA B" en `src/03_modelo_bayesiano.py`, `informe/secciones/04_inferencia.tex`, `08_apendice.tex`

### Parte 1 — Correcciones de bayesiana
1. [ ] Releer la primera entrega de bayesiana y las correcciones de `docs/correcciones.md`.
2. [ ] Escribir por qué la variable objetivo pide un modelo **ordinal** y qué supuestos trae ese modelo.
3. [ ] Escribir la **likelihood** del modelo con notación formal.
4. [ ] Revisar cada prior de la entrega anterior y decidir cuáles se quedan, cuáles se eliminan y por qué (¿la variable existe en la base?, ¿el parámetro existe en el modelo ordinal?).
5. [ ] Para cada prior que quede: distribución, hiperparámetros y argumento con fuente.
6. [ ] Revisar los odds ratios citados: ¿a qué tipo de variable y a qué escala se refieren?, ¿cómo se trasladan a la escala de los coeficientes del modelo?
7. [ ] Acordar con la Persona C la escala de las variables (depende del preprocesamiento).

### Parte 2 — Código (sección "PERSONA B")
8. [ ] Implementar `log_likelihood`, `log_prior` y `log_posterior`.
9. [ ] Probar las funciones con valores sencillos donde se pueda verificar a mano el resultado.
10. [ ] Graficar las priors elegidas para incluirlas en el informe.
11. [ ] Planear un análisis de sensibilidad: qué pasa con la posterior si las priors cambian.

### Parte 3 — Informe
12. [ ] Sección de metodología de inferencia bayesiana: justificación, likelihood, priors y posterior.
13. [ ] Apéndice matemático (10 % de la nota): derivaciones que respalden lo anterior.

### Parte 4 — Presentación
14. [ ] Diapositivas del modelo, la likelihood y las priors.

### Parte 5 — Revisión cruzada
15. [ ] Revisar el Pull Request de la Persona C.

---

## Persona C — Juan Esteban · Preprocesamiento, MCMC y resultados

**Archivos:** `src/02_preprocesamiento.py`, sección "PERSONA C" de `src/03_modelo_bayesiano.py`, `src/04_convergencia.py`, `src/05_evaluacion.py`, `informe/secciones/05_modelacion.tex`, `06_resultados.tex`

### Parte 1 — Preprocesamiento
1. [ ] Con los hallazgos de la Persona A, decidir el tratamiento de cada problema de calidad.
2. [ ] Decidir cómo partir en entrenamiento y prueba evitando fuga de información.
3. [ ] Decidir la transformación de las variables y acordarla con la Persona B.
4. [ ] Completar `02_preprocesamiento.py` y documentar cada decisión.

### Parte 2 — Algoritmo y MCMC
5. [ ] Elegir entre Metropolis-Hastings y Gibbs y justificar la elección.
6. [ ] Escribir el algoritmo en pseudocódigo para el informe.
7. [ ] Implementar `metropolis_hastings` y el bloque principal, usando las funciones de la Persona B.
8. [ ] Definir número de cadenas, iteraciones, burn-in y valores iniciales, y justificarlos.

### Parte 3 — Convergencia (`04_convergencia.py`)
9. [ ] Producir los gráficos de convergencia y los diagnósticos numéricos vistos en clase.
10. [ ] Si no hay convergencia, revisar con la Persona B (ver el diagrama del enunciado) antes de seguir.

### Parte 4 — Validación (`05_evaluacion.py`)
11. [ ] Obtener predicciones para la base de prueba a partir de la posterior.
12. [ ] Definir la regla de clasificación, conectándola con la gravedad de los falsos negativos.
13. [ ] Elegir y calcular las métricas de desempeño, con su incertidumbre.
14. [ ] Cuantificar el peso predictivo de cada variable.

### Parte 5 — Informe
15. [ ] Sección de metodología de modelación bayesiana.
16. [ ] Resultados y discusión: tablas y figuras del modelo (no exploratorias), en tiempo pasado, y respuesta explícita a la pregunta de investigación.

### Parte 6 — Presentación
17. [ ] Diapositivas de resultados.

### Parte 7 — Revisión cruzada
18. [ ] Revisar el Pull Request de la Persona A.

---

## Cierre — Todos

1. [ ] Reunión para escribir las **conclusiones** y la discusión final.
2. [ ] Verificar que todas las casillas de `docs/correcciones.md` estén marcadas.
3. [ ] Correr los scripts en orden desde cero en un computador distinto al de quien los escribió.
4. [ ] Compilar el PDF y revisar el límite de 15 páginas.
5. [ ] Ensayar la presentación completa.
6. [ ] Nombrar los archivos `GrupoX` y subir a Teams.

---

## Cronograma sugerido

| Semana | Persona A | Persona B | Persona C |
|---|---|---|---|
| 1 | Pasos 1–5 y EDA (9–11) | Pasos 1–7 | Preprocesamiento (1–4) |
| 2 | Pasos 6–8 y 12–13 | Código (8–11) | Algoritmo y MCMC (5–10) |
| 3 | Presentación, revisión de B | Informe, apéndice, revisión de C | Validación e informe, revisión de A |
| Final | Cierre en grupo | | |

**Dependencias:** C necesita el EDA de A para preprocesar; C necesita las funciones de B para correr el MCMC; A necesita los resultados de C para el resumen.

---

## Reglas de trabajo en Git

1. Nadie trabaja directo en `main`; cada tarea va en su rama (`nombre/tarea`).
2. Commits pequeños con mensajes que digan qué cambió.
3. Antes de empezar cada día: `git checkout main`, `git pull` y luego `git merge main` en su rama.
4. Al terminar una tarea: `git push` y abrir un Pull Request para el revisor asignado.
5. Cada persona edita solo sus archivos de `informe/secciones/` para evitar conflictos.
6. No subir archivos generados pesados (cadenas `.npy`); se regeneran con los scripts.
