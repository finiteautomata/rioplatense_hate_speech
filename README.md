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

Si estamos en entorno AMD RocM

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
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

## VLLM

Experimental

Nota: no pude hacerlo andar porque no tiene 8 bit

Para hacer que `cupy` funcione...
```bash
pip install cupy-cuda12x==12.1
python -m cupyx.tools.install_library --library nccl --cuda 12.x
```

## Nota sobre datasets

Hay un script que se llama `bin/split_dataset.py` que se encarga de partir un split en varios pedacitos más chiquitos. Esto es para poder usar varias claves de OpenAI en simultáneo y no esperar tanto tiempo.

Como resultado para las distintas predicciones tenemos

- `data/test_0k.csv`: Split k-ésimo del dataset de test sin ninguna predicción
- `data/test_0k_beto.csv`: Predicciones con BETO
- `data/test_0k_pred_12shot.csv`: Predicciones con ChatGPT + CoT 12 shot
- `data/test_0k_pred_1shot.csv`: Predicciones con ChatGPT + CoT 1 shot

Hay otros archivos con diferentes prompts que utilizamos

- `data/sample_dev_df.csv`
- `data/sample_dev_df_13_shot.csv`
- `data/sample_dev_df_14_shot.csv`
- `data/sample_dev_df_14_shot_tercera_posicion.csv`: acá agregamos un "no queda claro
