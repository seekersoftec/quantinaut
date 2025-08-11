.PHONY: install run clean

install:
	@echo "=== Installing NautilusAI ==="
	uv venv
	. .venv/bin/activate && uv pip install -e .
	@echo "=== Installation complete ==="

run:
	@echo "=== Running NautilusAI ==="
	. .venv/bin/activate && python -m nautilus_ai

clean:
	@echo "=== Removing virtual environment ==="
	rm -rf .venv


# 
# make install   # Creates venv and installs dependencies
# make run       # Activates venv and runs NautilusAI
# make clean     # Deletes venv
