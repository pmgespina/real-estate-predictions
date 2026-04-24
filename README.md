# Real-Estate Image Classifier

Automatic classification of real-estate listing images into 15 scene categories using transfer learning on PyTorch. Experimentation tracked with Weights & Biases. Production inference served via FastAPI + Streamlit.

**Authors:** Pilar Garzón · Pablo Moreno · Carolina Macho · Sofía Sebastián  
**Institution:** ICAI — Universidad Pontificia Comillas  
**Course:** Machine Learning II — Deep Learning, 2026

---

## Links

- **W&B workspace:** [202514287-universidad-pontificia-comillas](https://wandb.ai/202514287-universidad-pontificia-comillas)

---

## Classes

Bedroom · Coast · Forest · Highway · Industrial · Inside city · Kitchen · Living room · Mountain · Office · Open country · Store · Street · Suburb · Tall building

---

## Project structure

```
├── cnn.py                        # Core module: CNN class, data loading and utilities
├── evaluate_model.py             # Evaluation: classification report + confusion matrix
├── experiments/
│   ├── screening.py              # Phase 1 — initial screening of 9 architectures
│   ├── optimize_densenet.py      # Phase 1b — dedicated fine-tuning for DenseNet121
│   ├── compare_cnn.py            # Phase 2 — high-capacity models with two-phase protocol
│   ├── opt_f1.py                 # Phase 3 — F1 optimisation with Weighted CrossEntropy
│   ├── tuning_resnet.py          # Phase 4 — Bayesian hyperparameter sweep (W&B Sweeps)
│   └── definitive_training.py   # Final production training with best hyperparameters
├── api/
│   ├── main.py                   # FastAPI inference backend
│   └── resnext101_32x8d_prod.pt  # Model checkpoint (place here)
├── streamlit/
│   └── app.py                    # Streamlit frontend
├── dataset/
│   ├── training/
│   └── validation/
└── README.md
```

---

## Setup and execution

### Option A — uv (recommended)

More information: https://github.com/astral-sh/uv

**1. Install uv:**
```bash
pip install uv
```

**2. Create the virtual environment:**
```bash
uv venv --python=3.12 dl_lab
```

**3. Activate the environment:**
```bash
source dl_lab/bin/activate   # macOS / Linux
dl_lab\Scripts\activate      # Windows
```

**4. Install dependencies:**
```bash
uv pip install -r requirements.txt
```

---

### Option B — conda

**1. Create the virtual environment:**
```bash
conda create -n dl_lab python=3.12
```

**2. Activate the environment:**
```bash
conda activate dl_lab
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

### 3. Place the model checkpoint

Make sure the file `resnext101_32x8d_prod.pt` is inside the `api/` folder, next to `main.py`.

### 4. Open two terminals

**Terminal 1 — API:**
```bash
uvicorn api.main:app --port 8080 --reload --reload-dir api
```
Wait until you see: `Model 'resnext101_32x8d' loaded on cpu`  
Do not close this terminal.

**Terminal 2 — Streamlit:**
```bash
streamlit run streamlit/app.py
```

The app will open automatically at `http://localhost:8501`.  
API documentation (Swagger UI) is available at `http://localhost:8080/docs`.

---



## Dataset structure

```
dataset/
├── training/
│   ├── Bedroom/
│   ├── Coast/
│   └── ...
└── validation/
    ├── Bedroom/
    ├── Coast/
    └── ...
```

---