# 🌾 Deep Coffee - Herramienta de Clasificación de Granos de Café

> **Una solución de inteligencia artificial para los agricultores de café salvadoreños**

Deep Coffee es un proyecto de aprendizaje profundo diseñado para ayudar a los productores de café salvadoreños a **clasificar automáticamente la calidad y estado de tostación de sus granos** mediante análisis de imágenes.

## 🎯 Propósito del Proyecto

El café salvadoreño es reconocido mundialmente por su calidad premium, pero la clasificación manual de granos es:
- **Lenta y laboriosa**
- **Inconsistente** entre clasificadores
- **Difícil de escalar** para pequeños y medianos productores

Deep Coffee resuelve esto usando **redes neuronales de aprendizaje profundo** para clasificar automáticamente los granos en **4 categorías de tostación**:

| Categoría | Descripción |
|-----------|------------|
| 🟫 **Oscuro** | Grano completamente tostado |
| 🟢 **Verde** | Grano sin tostar (crudo) |
| ☕ **Claro** | Tostación ligera |
| 🟠 **Medio** | Tostación equilibrada |

## 🚀 Objetivo

Construir una herramienta confiable y accesible que:
1. **Mejore la calidad** del café salvadoreño mediante clasificación consistente
2. **Aumente la productividad** de los cafetales
3. **Reduzca costos operativos** de clasificación manual
4. **Facilite la trazabilidad** de lotes de café
5. **Empodera agricultores** con tecnología moderna

## 📁 Estructura del Proyecto

```
deep_coffee_project/
├── configs/                    # ⚙️ Configuración centralizada
├── data/                       # 📊 Conjuntos de datos (raw, procesado)
├── figures/                    # 📈 Gráficos y visualizaciones
├── src/                        # 💻 Código fuente principal
│   ├── data/                   #   • Carga y procesamiento de datos
│   ├── models/                 #   • Arquitecturas de redes neuronales
│   ├── training/               #   • Loops de entrenamiento
│   ├── evaluation/             #   • Métricas y evaluación
│   └── utils/                  #   • Utilidades y helpers
├── scripts/                    # 🔧 Scripts para entrenar y evaluar
├── notebooks/                  # 📓 Análisis exploratorio (Jupyter)
├── results/                    # 📋 Resultados de entrenamientos
├── Makefile                    # ⚡ Atajos de desarrollo
├── pyproject.toml              # 📦 Metadatos del proyecto
└── requirements.txt            # 📚 Dependencias Python
```

## 💾 Instalación

### 1) Crear ambiente virtual (Python 3.9+)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Instalar dependencias

```bash
pip install -r requirements.txt
```

O usando Make:

```bash
make install
```

## 📊 Estructura de Datos Esperada

Coloca tu conjunto de datos en:

```text
data/raw/coffee_beans/
├── train/
│   ├── dark/       (Imágenes de granos oscuros)
│   ├── green/      (Imágenes de granos verdes)
│   ├── light/      (Imágenes de granos claros)
│   └── medium/     (Imágenes de granos medio)
└── test/
    ├── dark/
    ├── green/
    ├── light/
    └── medium/
```

**Formatos soportados:** `.jpg`, `.jpeg`, `.png`

## 🏋️ Etapa 1: Entrenar Modelo Base (MLP)

```bash
python scripts/train_stage1.py
```

o con Make:

```bash
make train
```

**Salidas generadas:**
- `models/stage1_mlp_best.pt` - Mejor modelo guardado
- `results/stage1_training_history.json` - Historial de entrenamiento
- `figures/stage1_training_curves.png` - Gráficos de pérdida y precisión

## 📊 Etapa 1: Evaluar en Conjunto de Prueba

```bash
python scripts/evaluate_stage1.py
```

o con Make:

```bash
make eval
```

**Salidas generadas:**
- `results/stage1_test_metrics.json` - Métricas en conjunto de prueba
- `figures/stage1_confusion_matrix.png` - Matriz de confusión

## 📈 Métricas Principales

- **Precisión (Accuracy)** - Porcentaje de clasificaciones correctas
- **F1-Score Ponderado** - Balance entre precisión y exhaustividad
- **F1-Score Macro** - Promedio de F1 por categoría
- **Reporte de Clasificación** - Detalle por clase
- **Matriz de Confusión** - Errores de clasificación
- **Curvas de Entrenamiento** - Progreso de pérdida y precisión

## 🗺️ Hoja de Ruta - 5 Etapas

### Etapa 1 (ACTUAL) ✅
**Modelo Base MLP + Análisis Exploratorio**
- Línea base de rendimiento
- Dataset documentado
- Reproducibilidad garantizada

### Etapa 2 🔄
**Redes Neuronales Convolucionales (CNN)**
- Arquitecturas especializadas para imágenes
- Mejor extracción de características visuales

### Etapa 3 📚
**Transfer Learning**
- Uso de modelos preentrenados (ResNet, VGG)
- Aprovecha conocimiento de millones de imágenes

### Etapa 4 🎨
**Componentes Generativos**
- Data augmentation automático
- Síntesis de nuevas imágenes de entrenamiento

### Etapa 5 🚀
**Fine-tuning y Deployment**
- Interfaz web (Gradio/Streamlit)
- Modelo listo para usar en campo

## 🔧 Buenas Prácticas de Ingeniería

✓ **Configuración centralizada** - Un solo archivo para todos los parámetros
✓ **Reproducibilidad garantizada** - Seeds determinísticos
✓ **Código modular** - Fácil de mantener y actualizar
✓ **Scripts limpios** - Puntos de entrada mínimos y claros
✓ **Version control** - Todo en Git con commits documentados

## 📦 Requisitos del Sistema

- Python 3.9 o superior
- PyTorch 2.0+
- scikit-learn, pandas, numpy
- PyYAML para configuración
- (Opcional) Jupyter para notebooks

Ver `requirements.txt` para versiones exactas.

## 🤝 Contribución y Desarrollo

Para colaboradores:

```bash
# Crear rama para nueva feature
git checkout -b feature/tu-nueva-feature

# Hacer cambios y commit
git add .
git commit -m "descripción clara del cambio"

# Push y crear Pull Request
git push origin feature/tu-nueva-feature
```

## 📧 Contacto y Soporte

¿Preguntas sobre Deep Coffee?
- Creador: Gerard Portillo
- Email: gerardoportillo@example.com
- Repository: https://github.com/gerardoportillo/deep_coffee_project

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo `LICENSE` para detalles.

---

**Deep Coffee** - *Mejorando la calidad del café salvadoreño con inteligencia artificial* ☕🌾

## Future Work

- Hyperparameter sweep infrastructure
- Model registry and experiment tracking
- Hugging Face model integration
- Deployment hardening with monitoring and CI/CD

## License

MIT License (see `LICENSE`).
