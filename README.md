# mlflow-template

Example Repository to experiment with Mlflow.

# XGBoost MacOS Setup Instructions

If you're encountering issues with XGBoost, particularly on macOS, follow these steps to resolve the problem:

## 1. Install OpenMP

XGBoost requires OpenMP, which can be installed using Homebrew. If you don't have Homebrew installed, install it first.

```bash
brew install libomp
```

## 2. Set Environment Variables

After installing libomp, you need to set some environment variables so that XGBoost can find it. Add these lines to your shell configuration file (e.g., `~/.zshrc` or `~/.bash_profile`):

```bash
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/opt/libomp/lib
export CPATH=$CPATH:/usr/local/opt/libomp/include
```

After adding these lines, restart your terminal or run:

```bash
source ~/.zshrc  # or the appropriate file you edited
```

## 3. Reinstall XGBoost

After setting up OpenMP, reinstall XGBoost:

```bash
pip uninstall xgboost
pip install xgboost
```

## Additional Notes

- If you continue to experience issues, you may need to install XGBoost from source or use a version without OpenMP. Refer to the XGBoost documentation for more advanced installation methods.

For more information, visit the [XGBoost Installation Guide](https://xgboost.readthedocs.io/en/latest/install.html).


## MLflow information

in order to run experiments through MLflow, it is necessary to run mlflow server

> mlflow server --backend-store-uri file:<path-to_mlflow-database> --port=<port_number>