# rioplatense_hate_speech

Repo de experimentos sobre diferencias HS rioplatense vs ibérico

## Instalación

0. Crear entorno virtual

```bash
python -m venv .venv
```

1. Activar entorno virtual

```bash
source .venv/bin/activate
```

2. Instalar dependencias

```bash
pip install -r requirements.txt
pip install -e .
```

3. Agregar API key de OpenAI

```bash
cp config/config.ini.template config/config.ini
```

y cambiar en `config/config.ini` el valor de `OPENAI.API_KEY` por la API key de OpenAI

## Predicción

```bash
python predict_dataframe.py --input <input_csv> --output <output_csv>
# Opcional: --model_name <model_name>
```
