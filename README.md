# FlameletMLP Project

## Description
This project trains a Multilayer Perceptron (MLP) to predict species mass fractions and production rates from flamelet manifold data.

## File Structure
- `main.py`       : Main training and evaluation script.
- `model.py`      : Defines the `FlameletMLP` model architecture.
- `model_data.csv`: Input dataset (flamelet manifold table).
- `predictions.csv`: Model predictions on the test set.
- `y_test.csv`    : Ground-truth targets for the test set.
- `species_errors.csv`: MAE and RMSE for selected species.
- `loss_curve.png`: Training vs. validation loss curve.
- `pred_vs_actual_<species>.png`: Predicted vs actual scatter plots for each species.

## Requirements
- Python 3.8+
- PyTorch with MPS support (for macOS)
- scikit-learn
- pandas
- numpy
- matplotlib
- tqdm

## Installation
1. Clone the repository:
   ```bash
   git clone https://your-repo-url.git
   cd your-repo
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch scikit-learn pandas numpy matplotlib tqdm
   ```

## Usage
1. Place `model_data.csv` in the project directory.
2. Run the training script:
   ```bash
   python main.py
   ```
3. Outputs will be saved:
   - `model_best.pth`, `model_final.pth`
   - `predictions.csv`, `y_test.csv`
   - `species_errors.csv`
   - Plot images (`.png`)

## Customization
- Adjust model hyperparameters in `main.py` (learning rate, batch size, epochs).
- Modify selected species in the `sel` list for error computation and plotting.

## License
This project is provided under the MIT License.
# FGM
