# Riesgo materno · Regresión logística ordinal bayesiana

¿En qué medida las variables fisiológicas registradas permiten clasificar con precisión el nivel de riesgo de una gestante, y cuáles de estos factores tienen mayor peso predictivo?

**Equipo:** Sara Gabriela Chisaba Cárdenas · Alejandra Cortes Murillo · Juan Esteban Silva Arévalo
**Cursos:** Pensamiento Crítico 2 y Estadística Bayesiana — Universidad Externado de Colombia, 2026

**Datos:** Ahmed, M. (2020). *Maternal Health Risk* [Dataset]. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/863/maternal+health+risk · DOI: 10.24432/C5DP5D

## Estructura

```
├── data/
│   ├── raw/maternal_health_risk.csv   # base original (no se modifica)
│   └── processed/                     # bases procesadas (las generan los scripts)
├── src/
│   ├── config.py                      # rutas y constantes compartidas
│   ├── 01_eda.py                      # Persona A
│   ├── 02_preprocesamiento.py         # Persona C
│   ├── 03_modelo_bayesiano.py         # Persona B (likelihood, priors) + C (MCMC)
│   ├── 04_convergencia.py             # Persona C
│   └── 05_evaluacion.py               # Persona C
├── notebooks/                         # pruebas y exploración libre
├── results/
│   ├── figures/                       # figuras para el informe
│   └── tables/                        # tablas para el informe
├── informe/
│   ├── main.tex
│   ├── secciones/                     # un .tex por sección, cada uno con su responsable
│   └── Bibliografia.bib
├── docs/
│   ├── Enunciado_Proyecto.pdf
│   ├── entregas_anteriores/
│   └── correcciones.md                # checklist de correcciones
├── GUIA_TRABAJO.md                    # paso a paso por persona
├── requirements.txt
└── .gitignore
```

Los scripts traen solo la estructura y los `TODO` que cada responsable debe completar (ver `GUIA_TRABAJO.md`).

## Orden de ejecución

```bash
pip install -r requirements.txt
cd src
python 01_eda.py
python 02_preprocesamiento.py
python 03_modelo_bayesiano.py
python 04_convergencia.py
python 05_evaluacion.py
```

## Conectar el repositorio (paso a paso)

### Una sola persona crea el repositorio
1. En github.com → **New repository** → nombre `riesgo-materno-bayesiano` → *Private* → **sin** README ni .gitignore (ya vienen en esta carpeta) → *Create repository*.
2. **Settings → Collaborators → Add people**: agregar los usuarios de GitHub de las otras dos personas.
3. Descomprimir esta carpeta, abrir una terminal dentro de ella y ejecutar:
   ```bash
   git init
   git add .
   git commit -m "Estructura inicial del proyecto"
   git branch -M main
   git remote add origin https://github.com/USUARIO/riesgo-materno-bayesiano.git
   git push -u origin main
   ```
   Si pide contraseña: GitHub ya no acepta la contraseña de la cuenta; crear un token en **Settings → Developer settings → Personal access tokens → Fine-grained**, o iniciar sesión con GitHub Desktop / VS Code.

### Las otras dos personas
1. Aceptar la invitación (llega al correo o en github.com/notifications).
2. Clonar:
   ```bash
   git clone https://github.com/USUARIO/riesgo-materno-bayesiano.git
   cd riesgo-materno-bayesiano
   pip install -r requirements.txt
   ```

### Flujo diario (todos)
```bash
git checkout main
git pull                              # traer lo último
git checkout -b nombre/tarea          # una rama por tarea
# ... trabajar ...
git add archivo1 archivo2
git commit -m "Describe el cambio"
git push -u origin nombre/tarea
```
Luego en GitHub: **Compare & pull request** → asignar revisor → *Merge* cuando esté aprobado.

Primera configuración en un computador nuevo:
```bash
git config --global user.name "Nombre Apellido"
git config --global user.email "correo@ejemplo.com"
```
