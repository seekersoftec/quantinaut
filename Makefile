.PHONY: install run clean

install:
	@echo "=== Installing Quantinaut ==="
	uv venv
	. .venv/bin/activate && uv pip install -e .
	@echo "=== Installation complete ==="

run:
	@echo "=== Running Quantinaut ==="
	. .venv/bin/activate && python -m quantinaut

clean:
	@echo "=== Removing virtual environment ==="
	rm -rf .venv


# 
# make install   # Creates venv and installs dependencies
# make run       # Activates venv and runs Quantinaut
# make clean     # Deletes venv
