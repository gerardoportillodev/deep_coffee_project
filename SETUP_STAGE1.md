# 🚀 Guía de Setup - Stage 1 Deep Coffee

## ✅ Estado Actual

| Componente | Estado | Acción |
|-----------|--------|--------|
| 📝 Scripts (train/eval) | ✅ LISTOS | - |
| 🤖 Código (modelos/data) | ✅ COMPLETO | - |
| 📦 Dependencias | ❌ FALTAN | Instalar |
| 📊 Dataset | ❌ VACÍO | Descargar |
| 🔧 Python | ✅ 3.12.7 OK | - |

---

## 🔧 PASO 1: Instalar Dependencias

```bash
# Opción A: Instalar todo desde requirements.txt
pip install -r requirements.txt

# Opción B: Instalar solo lo esencial para Stage 1
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy pillow pyyaml tqdm

# Verificar instalación
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
```

**⏱️ Tiempo:** ~5-10 minutos (depende de tu conexión)

---

## 📊 PASO 2: Descargar Dataset

Tienes 2 opciones:

### Opción A: Descargar desde Kaggle (Automático)

```bash
# 1. Primero, configura credenciales Kaggle
mkdir -p ~/.kaggle
# Descarga kaggle.json desde: https://www.kaggle.com/account/api
# Colócalo en ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. Descargar dataset
mkdir -p data/external
kaggle datasets download -d gpiosenka/coffee-bean-dataset-resized-224-x-224 -p data/external
unzip data/external/coffee-bean-dataset-resized-224-x-224.zip -d data/external/coffee_beans

# 3. Preparar estructura esperada
mkdir -p data/raw/coffee_beans/{train,test}/{dark,green,light,medium}
# (Mover archivos según estructura)
```

### Opción B: Dataset Local (Si ya lo tienes)

```bash
# Coloca tus datos en:
data/raw/coffee_beans/
├── train/
│   ├── dark/     (imágenes aquí)
│   ├── green/
│   ├── light/
│   └── medium/
└── test/
    ├── dark/
    ├── green/
    ├── light/
    └── medium/
```

---

## 🏋️ PASO 3: Entrenar Modelo (Etapa 1)

Una vez instaladas dependencias y datos listos:

```bash
# Opción A: Comando directo
python scripts/train_stage1.py

# Opción B: Usando Make
make train

# Opción C: Desde Jupyter
jupyter notebook notebooks/01_stage1_eda_and_mlp.ipynb
```

**⏱️ Tiempo:** ~5-15 minutos (depende de GPU/CPU)

**Salidas esperadas:**
```
✓ models/stage1_mlp_best.pt
✓ results/stage1_training_history.json
✓ figures/stage1_training_curves.png
```

---

## 📊 PASO 4: Evaluar en Test Set

```bash
# Opción A: Comando directo
python scripts/evaluate_stage1.py

# Opción B: Usando Make
make eval
```

**Salidas esperadas:**
```
✓ results/stage1_test_metrics.json
✓ figures/stage1_confusion_matrix.png
```

---

## 💾 Estructura Final Esperada

```
data/raw/coffee_beans/
├── train/
│   ├── dark/       (N imágenes)
│   ├── green/      (N imágenes)
│   ├── light/      (N imágenes)
│   └── medium/     (N imágenes)
└── test/
    ├── dark/       (N imágenes)
    ├── green/      (N imágenes)
    ├── light/      (N imágenes)
    └── medium/     (N imágenes)

results/
├── stage1_training_history.json
└── stage1_test_metrics.json

figures/
├── stage1_training_curves.png
└── stage1_confusion_matrix.png

models/
└── stage1_mlp_best.pt
```

---

## ✅ Checklist Completo

- [ ] Instalar `requirements.txt`
- [ ] Descargar/preparar dataset en `data/raw/coffee_beans/`
- [ ] Verificar estructura de carpetas con `python -c "from src.data.dataset import get_dataloaders; print('✓ Dataset OK')"`
- [ ] Ejecutar `python scripts/train_stage1.py`
- [ ] Revisar métricas en `results/stage1_test_metrics.json`
- [ ] Ejecutar `python scripts/evaluate_stage1.py`
- [ ] Visualizar `figures/stage1_training_curves.png`

---

## 🆘 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'torch'`
```bash
pip install torch torchvision
```

### Error: `FileNotFoundError: No images found in data/raw/coffee_beans/train`
```bash
# Verifica que tienes imágenes en:
ls data/raw/coffee_beans/train/dark/
# Debe haber imágenes .jpg, .png aquí
```

### Error: `CUDA out of memory`
```bash
# Edita config.py y reduce batch_size
# batch_size: int = 16  # de 32 a 16
```

---

## 📖 Próximos Pasos (Después de Stage 1)

1. Revisar métricas en `results/stage1_test_metrics.json`
2. Analizar matriz de confusión
3. Ajustar hiperparámetros si es necesario
4. Pasar a **Stage 2: CNN** (cuando esté listo)

---

**¿Listo para ejecutar? Comienza con el PASO 1️⃣ 🚀**
