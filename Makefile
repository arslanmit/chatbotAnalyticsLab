.PHONY: help demo train test verify batch optimize main clean install check-deps setup-venv

# Python virtual environment
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3

# Default target
help:
	@echo "Chatbot Analytics & Optimization - Available Commands"
	@echo "======================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup-venv       - Create Python virtual environment"
	@echo "  make install          - Install Python dependencies"
	@echo "  make install-gpu      - Install dependencies with GPU support (CUDA 11.8)"
	@echo "  make check-deps       - Check if dependencies are installed"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make demo             - Run dataset pipeline demo (load, validate, preprocess)"
	@echo ""
	@echo "Intent Classification:"
	@echo "  make train            - Train BERT-based intent classifier"
	@echo "  make test             - Test trained model on sample queries"
	@echo "  make verify           - Comprehensive model evaluation with metrics"
	@echo "  make batch            - Batch processing & optimization demo"
	@echo ""
	@echo "Application:"
	@echo "  make main             - Run main application"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Remove cache files and temporary data"
	@echo "  make clean-models     - Remove trained models"
	@echo ""

# Setup commands
setup-venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "Virtual environment created! Activate with: source venv/bin/activate"

install: setup-venv
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed!"

install-gpu: setup-venv
	@echo "Installing dependencies with GPU support (CUDA 11.8)..."
	$(PIP) install -r requirements.txt
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu118
	@echo "Dependencies with GPU support installed!"

check-deps:
	@echo "Checking dependencies..."
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@$(PYTHON) -c "import datasets; print(f'Datasets: {datasets.__version__}')"
	@$(PYTHON) -c "import pandas; print(f'Pandas: {pandas.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Data pipeline
demo:
	@echo "Running dataset pipeline demo..."
	PYTHONPATH=. $(PYTHON) examples/dataset_pipeline_demo.py

# Intent classification
train:
	@echo "Training intent classifier..."
	$(PYTHON) examples/train_intent_classifier.py

train-quick:
	@echo "Quick training (1 epoch, CPU-only)..."
	$(PYTHON) examples/train_intent_classifier_quick.py

test:
	@echo "Testing intent classifier..."
	$(PYTHON) examples/test_intent_classifier_basic.py

verify:
	@echo "Verifying intent classifier..."
	$(PYTHON) examples/verify_intent_classifier.py

batch:
	@echo "Running batch optimization demo..."
	$(PYTHON) examples/test_batch_optimization.py

optimize: batch

# Main application
main:
	@echo "Running main application..."
	PYTHONPATH=. $(PYTHON) src/main.py

# Cleanup
clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Cache cleaned!"

clean-models:
	@echo "Removing trained models..."
	rm -rf models/
	@echo "Models removed!"

# Quick workflow shortcuts
quick-start: demo train test
	@echo "Quick start complete!"

full-pipeline: demo train-quick test
	@echo "Full pipeline complete!"

full-pipeline-complete: demo train verify batch
	@echo "Complete pipeline with full training done!"
