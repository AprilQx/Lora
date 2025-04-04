Installation Guide
================

Requirements
----------

* Python 3.9+
* PyTorch 2.6+
* CUDA-compatible GPU (recommended)
* 16GB+ RAM

Quick Start
---------

1. Clone the repository:

.. code-block:: bash

   git clone https://gitlab.com/your-username/M2_coursework.git
   cd M2_coursework

2. Create and activate a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Data Setup
--------


1. Preprocess the data:

.. code-block:: bash

   python src/data/process_data.py



Project Structure
--------------

.. code-block:: text

   M2_coursework/
   ├── data/                  # Data files
   ├── docs/                  # Documentation
   ├── experiments/           # Experiment scripts
   │   ├── eval/              # Evaluation scripts
   │   ├── finetune/          # Fine-tuning scripts
   │   └── hyperparameter/    # Hyperparameter search
   ├── notebooks/             # Jupyter notebooks
   ├── results/               # Results and visualizations
   ├── src/                   # Source code
   │   ├── data/              # Data processing
   │   ├── evaluation/        # Evaluation utilities
   │   └── models/            # Model implementations
   └── utils/                 # Utility functions