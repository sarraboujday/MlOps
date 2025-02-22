VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
FLAKE8=$(VENV)/bin/flake8
BLACK=$(VENV)/bin/black

install:
	@echo "📦 Installation des dépendances..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	@echo "🔍 Vérification du code avec flake8..."
	$(FLAKE8) main.py model_pipeline.py

format:
	@echo "🎨 Formatage automatique du code avec Black..."
	$(BLACK) main.py model_pipeline.py

prepare:
	@echo "📊 Préparation des données..."
	@$(PYTHON) main.py --prepare

train:
	@echo "🚀 Entraînement du modèle..."
	@$(PYTHON) main.py --train --save

evaluate:
	@echo "📊 Évaluation du modèle..."
	@$(PYTHON) main.py --evaluate --load

all: train evaluate
	@echo "💥 Entraînement et évaluation terminés!"