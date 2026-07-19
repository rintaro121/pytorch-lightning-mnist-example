# PyTorch Lightning MNIST Example

Simple example of training a MLP classifier on MNIST with PyTorch Lightning.  
Training metrics and artifacts are logged with MLflow.

## Run Training

Install the dependencies and run the training script:

```bash
uv sync
uv run python src/train.py
```

Training results are written under `outputs/`:

## View Results

Start the MLflow UI from the project root:

```bash
uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

Select the `mnist-classification` experiment to view training, validation, and
test metrics.
