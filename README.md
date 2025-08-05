# DocShield: Document Authentication via Deep Learning

## Overview

DocShield is an end‑to‑end system for detecting fraudulent documents such as Social Security numbers (SSNs), driver’s licenses (DLs) and bank statements.  It includes data preparation, model training, hyperparameter tuning, evaluation, export, an inference API and a simple web UI.  The project is designed for practitioners who need a reproducible and extensible pipeline for document classification and explainability.

```
docshield/
  README.md               # This file
  LICENSE                 # Project license
  .gitignore              # Files to ignore in git
  pyproject.toml          # Python packaging metadata
  requirements.txt        # Python dependencies
  docker/                 # Dockerfiles for different services
    Dockerfile.api
    Dockerfile.ui
    Dockerfile.train
  docker-compose.yml      # Compose services: API and UI
  configs/                # YAML configuration files
    train.yaml
    model.yaml
    inference.yaml
    augmentations.yaml
  src/                    # Source code package
    docshield/
      __init__.py
      data/
        datasets.py
        transforms.py
        pdf_to_image.py
        synth_dummy_data.py
      models/
        factory.py
        efficientnet.py
        vit.py
        heads.py
      train/
        train.py
        eval.py
        tune.py
        utils.py
        metrics.py
      infer/
        service.py
        explain.py
        schemas.py
      api/
        main.py
      ui/
        app.py
      utils/
        logging.py
        config.py
  tests/                  # Unit tests
    test_datasets.py
    test_inference.py
    test_api.py
    test_transforms.py
  scripts/
    export_onnx.py
    download_models.py
    benchmark_infer.py
  models/                 # Saved model checkpoints (empty by default)
    .gitkeep
  data/                   # Data directory (empty by default)
    .gitkeep
  .github/workflows/ci.yml
  Makefile                # Common tasks
```

## Architecture

The system comprises three major components:

1. **Training and Evaluation** – Training scripts (under `src/docshield/train`) build and evaluate deep learning models using transfer learning.  They support EfficientNet‑B0 and Vision Transformer (ViT) backbones and handle class imbalance via weighted sampling.  Hyperparameter tuning is driven by Optuna to search over learning rate, weight decay, batch size and augmentation strength.
2. **Inference API** – A FastAPI application (under `src/docshield/api`) exposes endpoints `/health`, `/predict` and `/version`.  The API loads a trained model, accepts image or PDF uploads, performs inference and returns the predicted label, confidence and Grad‑CAM based saliency maps.  File processing happens entirely in memory by default, ensuring that no PII is stored on disk unless explicitly configured.
3. **Web UI** – A Streamlit application (under `src/docshield/ui`) offers a drag‑and‑drop interface for document classification.  Users upload an image or single‑page PDF, view the predicted label and confidence score, and inspect heatmaps showing which regions influenced the prediction.  An admin sidebar allows switching models, adjusting thresholds and selecting top‑k outputs.

### Data Flow

1. **Data Preparation** – The `synth_dummy_data.py` script generates a tiny synthetic dataset with obvious watermarks to validate the pipeline end‑to‑end.  Real datasets should follow the same directory structure: `data/train/<class_name>` and `data/val/<class_name>` with image (and PDF) files.  The `DocClassificationDataset` class reads these files and applies augmentations defined in `augmentations.yaml` via Albumentations.
2. **Training** – The `train.py` script loads configuration from `train.yaml` and `model.yaml`, instantiates the dataset and dataloader with balanced sampling, builds the model through `models.factory.create_model`, and trains using AdamW with a cosine annealing scheduler.  Metrics are logged to TensorBoard and checkpoints saved in `models/`.  Early stopping monitors macro‑F1 on the validation set.
3. **Evaluation & Tuning** – The `eval.py` script evaluates a saved model on a validation or test set, computing accuracy, precision, recall, F1 (macro and per class) and AUROC.  Confusion matrices and precision–recall curves are stored in `reports/`.  The `tune.py` script performs hyperparameter search via Optuna and writes the best configuration and plots into `tuning/`.
4. **Inference** – In production, the inference API loads the exported model and configuration from `inference.yaml`.  When a user uploads a file, the API converts PDFs to images (see `pdf_to_image.py`), preprocesses the image using the same transforms as training, runs inference, applies softmax to obtain probabilities and returns the top‑k results and Grad‑CAM saliency overlay.
5. **UI** – The Streamlit UI communicates with the API using HTTP.  It displays the uploaded image/PDF preview, the predicted class (e.g., `ssn_real`, `dl_fake`, etc.), confidence score and saliency map.  The admin sidebar includes model selection, threshold and top‑k controls.

## Setup

### Prerequisites

* Python 3.11
* CUDA‑enabled GPU (optional but recommended)
* Docker and Docker Compose (if running containerized)

### Installation (Bare Metal)

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Some optional dependencies such as `pdf2image` or `PyMuPDF` are required to convert PDFs to images.  Install them along with system packages (e.g. poppler) if your dataset contains PDFs.

### Running Training

First, generate or prepare your dataset under `data/train` and `data/val`.  For an end‑to‑end test with synthetic data, run:

```bash
python -m src.docshield.data.synth_dummy_data --output_dir data/train --num_samples 10
python -m src.docshield.data.synth_dummy_data --output_dir data/val --num_samples 5
```

Then launch training:

```bash
python -m src.docshield.train.train --config configs/train.yaml
```

You can override any configuration value via command line using Hydra syntax, for example:

```bash
python -m src.docshield.train.train --config configs/train.yaml optim.lr=1e-4 model.name=vit
```

### Hyperparameter Tuning

Run the tuning script to search over learning rate, weight decay, batch size and augmentation strength:

```bash
python -m src.docshield.train.tune --config configs/train.yaml
```

The best parameters and study plots are saved in `tuning/`.

### Evaluation

Evaluate a trained model checkpoint:

```bash
python -m src.docshield.train.eval --config configs/train.yaml --checkpoint models/best.ckpt
```

Evaluation metrics (accuracy, F1, AUROC, confusion matrix) are written to `reports/`.

### Exporting to ONNX

Export a trained model to ONNX format:

```bash
python scripts/export_onnx.py --checkpoint models/best.ckpt --output models/best.onnx --config configs/model.yaml
```

This produces an ONNX file that can be used for inference outside PyTorch.

### Running the API and UI

#### Using Docker Compose

To run the inference API and Streamlit UI via Docker Compose:

```bash
docker-compose up --build
```

This builds the API and UI images, starts the services and exposes:

* API at http://localhost:8000
* UI at http://localhost:8501

#### Bare Metal

Start the API server:

```bash
uvicorn src.docshield.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then in a separate terminal run the Streamlit UI:

```bash
streamlit run src/docshield/ui/app.py
```

### API Usage

Upload an image via `curl`:

```bash
curl -X POST -F file=@path/to/document.jpg http://localhost:8000/predict
```

Example response:

```json
{
  "document_type": "dl",
  "label": "real",
  "confidence": 0.983,
  "top_k": [
    {"label": "dl_real", "prob": 0.983},
    {"label": "dl_fake", "prob": 0.017}
  ],
  "explanations": {"saliency_png_base64": "..."},
  "model_version": "v1.0.0"
}
```

### Adding a New Document Type

1. Add two new classes to your dataset: `<doc>_real` and `<doc>_fake`.
2. Update `configs/model.yaml` to set `num_classes` accordingly.
3. Optionally extend `synth_dummy_data.py` to generate synthetic examples for the new document type.
4. Retrain the model.

### Performance Tips

* Use mixed precision training (`torch.cuda.amp`) to improve throughput on GPUs.
* Increase `num_workers` in dataloaders to parallelize image loading.
* Adjust image size: larger resolutions capture more detail but reduce batch size.
* For inference on CPU, export to ONNX and use optimized runtimes like ONNX Runtime.

### Security and PII

DocShield is designed to respect user privacy:

* **No persistence by default** – uploaded files are processed entirely in memory and not stored.  If disk persistence is enabled via configuration, choose a secure temporary directory and enforce retention policies.
* **Redacted logs** – do not log raw content of documents.  Log only anonymized metadata (e.g., file size, type, prediction result).
* **HTTPS recommended** – deploy behind an HTTPS reverse proxy to protect data in transit.
* **Third‑party assets** – clearly license any third‑party pretrained weights or datasets used for training.  This repository includes only synthetic data; replace with your own data responsibly.

### Troubleshooting

| Issue | Possible Cause | Fix |
|------|----------------|-----|
| `torch not installed` error | PyTorch is not installed in your environment | Install PyTorch via pip or conda according to your hardware and CUDA version. |
| `pdf2image` ImportError | PDF conversion is required but the library is missing | Install `pdf2image` and system dependencies (poppler).  Alternatively install `PyMuPDF` (fitz) and adjust `pdf_to_image.py`. |
| CUDA out of memory during training | Batch size or image size too large | Reduce `batch_size` or `image_size` in `configs/train.yaml`.  Enable gradient checkpointing for ViT models. |
| Inference API returns low confidence for all classes | Poorly trained model or domain shift | Ensure your training dataset covers the expected document distribution.  Perform hyperparameter tuning and increase dataset size. |

For further assistance please open an issue on the project repository.