# Multimodal RAG System

A research-grade Retrieval-Augmented Generation (RAG) system with multimodal support.

---

## Environment Setup

### Python Version Requirement

Python **3.10 or higher** is required. Verify with:

```
python --version
```

---

### 1. Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

### 2. Install Dependencies

```powershell
pip install torch==2.2.2 transformers==4.40.2 sentence-transformers==2.7.0 faiss-cpu==1.8.0 whoosh==2.7.4 numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 tqdm==4.66.4 matplotlib==3.8.4 pillow==10.3.0
```

Or install from the frozen requirements file:

```powershell
pip install -r requirements.txt
```

---

### 3. Freeze Dependencies

```powershell
pip freeze > requirements.txt
```

---

### GPU vs CPU Notes

| Environment | FAISS Package | Notes |
|---|---|---|
| Local / Windows | `faiss-cpu==1.8.0` | CPU-only; no CUDA required; suitable for development and testing |
| Kaggle / College GPU | `faiss-gpu` | GPU-accelerated; requires CUDA; significantly faster for large-scale vector search |

When deploying on a GPU environment (Kaggle, university HPC, or cloud GPU), replace `faiss-cpu` with the appropriate `faiss-gpu` build matching your CUDA version:

```powershell
pip uninstall faiss-cpu
pip install faiss-gpu
```
