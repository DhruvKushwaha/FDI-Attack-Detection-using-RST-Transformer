# FDI-Attack-Detection-using-RST-Transformer
This repository contains false data injection (FDI) attack methods for IEEE Test cases. 
This repository contains the source code and documentation for FDI-Attack-Detection-using-RST-Transformer. The repo achieves FDI attack detection for IEEE 18, 30 and 118 bus cases by leveraging the numerical computing power of MATLAB for MATPOWER-based simulations to generate data and the Python for implementing Deep learning architectures.

# 🛠️ Installation
Follow these steps to set up the project environment.

# Prerequisites
MATLAB version R2020b Matpower 8.1.

Python version 3.12 

requirements.txt file is included.

# Setup Instructions
Clone the repository:
```git
git clone https://github.com/DhruvKushwaha/FDI-Attack-Detection-using-RST-Transformer.git

cd FDI-Attack-Detection-using-RST-Transformer
```
## Set up the Python Environment:
It is highly recommended to use a virtual environment to manage dependencies.

Create a virtual environment:
```
python -m venv venv
```
Activate it (on Windows)
```
venv\Scripts\activate
```

Activate it (on macOS/Linux)
```
source venv/bin/activate
```
## Install the required packages
```
pip install -r requirements.txt
```
## Set up the MATLAB Environment:
Open MATLAB.

Navigate to the repository's root directory.

Add the necessary folders to the MATLAB path by running the following command in the MATLAB Command Window:
```matlab
addpath(genpath(fullfile(pwd, 'matlab')));
```
# 🚀 Usage
Instructions on how to run code.

## MATLAB Scripts
To run the main MATLAB simulation or analysis, execute the primary script from the MATLAB Command Window:

Ensure you are in the project's root directory
```
run('MATLAB/<Single/multi-bus attack>_<IEEE test case>.m') %to generate dataset for each case
run('MATLAB/Jacobian Sensitivity/<desired case>') %to find vulnerable buses in each case
```

## Python Scripts
To run the Python portion of the project, use each of the jupyter notebooks to execute the commands (with the virtual environment activated).

**Note: The datasets generated from MATLAB test cases must be copied in csv format to the respective cases in folders.**

# 🤝 Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/AmazingFeature).

Make your changes and commit them (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

# 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
