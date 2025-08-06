# Installation Guide

This guide outlines methods for installing and running nautilus_ai:

## Local Installation

This method installs NautilusAI directly on your machine.

**Prerequisites:**

- Git
- Python 3.11+

**Steps:**

1. **Clone the NautilusAI Repository:**

   ```bash
   git clone https://github.com/seekersoftec/nautilus_ai.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd nautilus_ai
   ```

3. **(Optional) Setup Environment:** You can choose from any of the following environment setups. First **install system dependencies:**

   ```bash
      sudo apt-get update

      sudo apt-get install -y build-essential clang wget curl git libbz2-dev python3-pip

      # sudo chmod +x scripts/install-talib.sh && bash scripts/install-talib.sh

      # sudo chmod +x scripts/install-pygame.sh && bash scripts/install-pygame.sh # (Optional)

      curl -sSL https://install.python-poetry.org | python -

   ```

   - **(Option A) Create a Virtual Environment:** A virtual environment helps isolate project dependencies. Here's an example using `venv`:

     ```bash
        python3 -m venv .venv  # Use a different venv manager if preferred
        source .venv/bin/activate
     ```

   - **(Option B) Create a Conda Environment:** A conda environment also helps isolate project dependencies. You can also control the version of python too:

     ```bash
     conda create --name nautilus_ai python=3.11  # Use the tested and recommended python version
     # using conda v4:
      conda activate nautilus_ai

      # OR

      # using conda v3:
      source activate nautilus_ai

      # Verify python version:
      python --version
     ```

4. **Install Dependencies:**

```bash
   poetry install # recommended package manager

   # OR

   pip install --upgrade pip
   pip install -e .
```
