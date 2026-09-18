# Predicting the Severity of Geomagnetic Disturbances Caused by Earth-impacting Coronal Mass Ejections Using Explainable Physics-aware Machine Learning

## Author

Hongyang Zhang

**Affiliation:** Institute for Space Weather Sciences and Department of Computer Science, New Jersey Institute of Technology

## Project Structure

- `trainingdata/`: Training and validation datasets containing 72 and 19 events, respectively, supplied for reference.
- `testing data/`: Test dataset containing 24 events.
- `model code/model.py`: Model loading and prediction functions.
- `pre-trained model and weights/`: Pretrained model bundle.
- `testing code/test.py`: Testing script for Max Kp prediction, evaluation, and Figure 3.
- `testing code/plot_figure3.py`: Standalone Figure 3 plotting script and plotting function.
- `workflow.ipynb`: Self-contained workflow with model code, testing code, and Figure 3 plotting code.
- `requirements.txt`: Required Python packages and their versions.
- `figure3.png`: Figure 3 generated from the test predictions.
- `prediction.pdf`: Printout of the executed notebook, including its code, results, and figure.

## Requirements

Tested with Python **3.13.9**. Main dependencies:

```txt
numpy==2.3.5
pandas==2.3.3
scipy==1.16.3
scikit-learn==1.8.0
quantile-forest==1.4.1
matplotlib==3.10.6
```

Install the complete dependency list from the repository root:

```bash
python -m pip install -r requirements.txt
```

The notebook also requires a local Jupyter installation using the same Python environment. Figure 3 uses Arial when available, with DejaVu Sans as a fallback.

## Testing

### Max Kp Prediction

Run testing from the repository root:

```bash
python "testing code/test.py"
```

The script prints each event's onset time, predicted Max Kp, prediction interval, and observed Max Kp, followed by MAE, MRE, R², and EC. Full-precision predictions and metrics are saved to `results/predictions.csv` and `results/metrics.json`. It also generates `figure3.png` in the repository root.

Expected results: **MAE = 0.8640**, **MRE = 0.1653**, **R² = 0.3832**, and **EC = 66.67%**.

### Figure 3

The plotting script can also be run directly, without running the test script first:

```bash
python "testing code/plot_figure3.py"
```

It computes fresh predictions from the supplied model and test data, saves the prediction files under `results/`, and generates `figure3.png` in the repository root.

To plot an existing prediction CSV instead:

```bash
python "testing code/plot_figure3.py" --predictions "results/predictions.csv"
```

Both scripts save the figure to a file. Open `figure3.png` to view it; the notebook displays it inline.

### Prediction Workflow

Open `workflow.ipynb` with the repository root as the working directory, select the Python environment with the required dependencies, and choose **Run All**. The notebook reads the local pretrained model and test data, runs the evaluation, and generates `figure3.png`. All required code is included in the notebook; it does not read external Python scripts.

The training and validation datasets are not used during testing. To update `prediction.pdf`, save the executed notebook and print it to PDF from Jupyter.

## Pretrained Models

The pretrained model is stored under:

```text
pre-trained model and weights/model.pkl
```

The bundle contains the fitted forest, its hyperparameters, feature order, and interval quantiles. Model loading code is included in `model code/model.py` and `workflow.ipynb`.

## Reference

- Zhang, H. *Predicting the Severity of Geomagnetic Disturbances Caused by Earth-impacting Coronal Mass Ejections Using Explainable Physics-aware Machine Learning*. Manuscript.
