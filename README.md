# Universal Optimal Transport Inversion (PyTorch)

This project implements an **optimal transport-based inversion framework** using PyTorch.  
Originally developed for **magnetotelluric (MT) geophysical inversion**, it can also be applied to any optimization task where point-set distances (1D–4D) are required.

## Features
- 🚀 Optimal transport loss with efficient implementation  
- 🔄 Supports 1D–4D point sets  
- ⚡ GPU-accelerated gradient descent with PyTorch  
- 🌍 Applications in geophysics (e.g., MT inversion)  
- 🔧 Extendable to general inverse problems

## Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
