# EasyLM

EasyLM is a Python library for linear regression with R-style summaries, automatic coefficient interpretation, and model comparison. It provides simple interfaces, comprehensive statistics, and built-in visualizations.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- R-like `lm()` interface with comprehensive summaries  
- Coefficients, standard errors, t-values, p-values, R², AIC, BIC  
- Compare multiple models side-by-side  
- Built-in plotting for model comparison  
- Accepts NumPy arrays, Pandas DataFrames, and Python lists  
- Clean, modular, and extensible architecture  

---

## Installation

Install directly from PyPI:

```bash
pip install amoang-easylm

Or install the latest version from GitHub:

```bash
git clone https://github.com/yourusername/EasyLM.git
cd EasyLM
pip install -r requirements.txt
pip install -e .


from EasyLM import LinearModel
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Fit model
model = LinearModel()
model.fit(X, y)

# View R-style summary
print(model.summary())

