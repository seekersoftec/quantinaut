import os
import platform
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, shell=False):
    """Run a command and stream output."""
    print(f"→ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    subprocess.check_call(cmd, shell=shell)

def main():
    print("=== NautilusAI Installer ===")

    auto_run = "--run" in sys.argv

    # Check uv
    try:
        subprocess.check_output(["uv", "--version"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  uv not found. Installing uv...")
        if platform.system() == "Windows":
            run_cmd([sys.executable, "-m", "pip", "install", "uv"])
        else:
            run_cmd("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)

    # Create venv
    run_cmd(["uv", "venv"])

    # Install dependencies
    print("📦 Installing NautilusAI dependencies...")
    run_cmd(["uv", "pip", "install", "-e", "."])

    # Paths for activation
    if platform.system() == "Windows":
        activate_path = Path(".venv") / "Scripts" / "activate"
        python_path = Path(".venv") / "Scripts" / "python"
    else:
        activate_path = Path(".venv") / "bin" / "activate"
        python_path = Path(".venv") / "bin" / "python"

    if auto_run:
        print("\n🚀 Running NautilusAI...")
        run_cmd([str(python_path), "-m", "nautilus_ai"])
        return

    print("\n🎉 Installation complete!")
    print(f"To activate the environment, run:\n  {activate_path}")
    print("To start NautilusAI, run:\n  python -m nautilus_ai")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)
        
"""
# Examples

**Install only**

```bash
python install.py
```

**Install + run immediately**

```bash
python install.py --run
```
"""