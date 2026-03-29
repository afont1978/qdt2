# Q-InfraTwin Web App

Repositorio preparado para GitHub y despliegue web de un prototipo **Quantum Digital Twin** basado en Streamlit.

## Qué incluye

- `app.py`: dashboard web listo para despliegue.
- `src/q_infratwin/engine.py`: motor híbrido cuántico-clásico modularizado.
- `requirements.txt`: dependencias mínimas.
- `Dockerfile`: despliegue contenedorizado.
- `.streamlit/config.toml`: configuración básica de Streamlit.
- `.gitignore`: exclusiones estándar para Python.

## Estructura

```text
q-infratwin-webapp/
├── app.py
├── requirements.txt
├── runtime.txt
├── Dockerfile
├── .gitignore
├── .streamlit/
│   └── config.toml
└── src/
    └── q_infratwin/
        ├── __init__.py
        └── engine.py
```

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Despliegue recomendado

### Opción 1: Streamlit Community Cloud
1. Sube este repo a GitHub.
2. En Streamlit Cloud, conecta el repositorio.
3. Selecciona `app.py` como archivo principal.
4. Despliega.

### Opción 2: Render / Railway / cualquier plataforma con Docker
Usa el `Dockerfile` incluido.

```bash
docker build -t q-infratwin-webapp .
docker run -p 8501:8501 q-infratwin-webapp
```

## Nota técnica

Este repo reemplaza la carga dinámica del motor por fichero con `exec()` por imports directos desde `src/q_infratwin`, lo que simplifica mantenimiento, versionado y despliegue.

## GitHub

Una vez creado tu repositorio vacío en GitHub:

```bash
git init
git add .
git commit -m "Initial commit: Q-InfraTwin web app"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```
