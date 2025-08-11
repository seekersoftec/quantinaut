# Installation Guide

This guide outlines methods for installing and running **nautilus\_ai** using [`uv`](https://github.com/astral-sh/uv), a fast Python package manager and environment tool.

---

## Prerequisites

* **Git**
* **Python 3.11+**
* **`uv` installed**

  * Install with pip:

    ```bash
    pip install uv
    ```
  * Or via shell script:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

---

## 1. Quick Install (Recommended)

If you want to clone the repo, create a venv, and install dependencies in one go:

**Linux / macOS**

```bash
git clone https://github.com/seekersoftec/nautilus_ai.git && \
cd nautilus_ai && \
uv venv && \
source .venv/bin/activate && \
uv pip install -e .
```

**Windows PowerShell**

```powershell
git clone https://github.com/seekersoftec/nautilus_ai.git; `
cd nautilus_ai; `
uv venv; `
.venv\Scripts\Activate.ps1; `
uv pip install -e .
```

After installation, run NautilusAI:

```bash
python -m nautilus_ai
```

---

## 2. Manual Installation

If you prefer more control over the installation process:

### Step 1 – Clone the Repository

```bash
git clone https://github.com/seekersoftec/nautilus_ai.git
cd nautilus_ai
```

### Step 2 – Install System Dependencies (Linux Example)

```bash
sudo apt-get update
sudo apt-get install -y build-essential clang wget curl git libbz2-dev python3-pip

# Optional: Install TA-Lib
# sudo chmod +x scripts/install-talib.sh && bash scripts/install-talib.sh

# Optional: Install Pygame
# sudo chmod +x scripts/install-pygame.sh && bash scripts/install-pygame.sh
```

### Step 3 – Setup a Virtual Environment

**Option A – uv-managed venv (Recommended)**

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

**Option B – Conda Environment**

```bash
conda create --name nautilus_ai python=3.11
conda activate nautilus_ai
python --version  # Verify Python version
```

**Option C – Python venv**

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

### Step 4 – Install Dependencies

```bash
uv pip install -e .
```

Or from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

