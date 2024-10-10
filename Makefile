SHELL=/bin/bash
# === USER PARAMETERS

ifdef OS
   export PYTHON_COMMAND=python
   export UV_INSTALL_CMD=pip install uv==0.2.26
   export VENV_BIN=.venv/Scripts
else
   export PYTHON_COMMAND=python3.12
   export UV_INSTALL_CMD=pip install uv==0.2.26
   export VENV_BIN=.venv/bin
endif

export SRC_DIR=test_project
ifndef BRANCH_NAME
	export BRANCH_NAME=$(shell git rev-parse --abbrev-ref HEAD)
endif
DEPLOY_ENVIRONMENT=$(shell if [ $(findstring main, $(BRANCH_NAME)) ]; then \
			echo 'prod'; \
		elif [ $(findstring pre, $(BRANCH_NAME)) ]; then \
			echo 'pre'; \
		else \
		 	echo 'dev'; \
		fi)
# If use deploy_environment in the tag system
# `y` => yes
# `n` => no
USE_DEPLOY_ENVIRONMENT=n

# == SETUP REPOSITORY AND DEPENDENCIES

install-uv:
	# install uv package manager
	$(UV_INSTALL_CMD)
	# create environment
	uv venv -p $(PYTHON_COMMAND)

set-hooks:
	cp .hooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
	cp .hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
	cp .hooks/post-merge .git/hooks/post-merge && chmod +x .git/hooks/post-merge

compile:
	# install extra dev group
	uv pip compile pyproject.toml --extra dev -o requirements.txt --cache-dir .uv_cache

install:
	uv pip sync requirements.txt --cache-dir .uv_cache
	
torch-gpu:
	uv pip install torch==2.4.0+cu124 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall

setup: install-uv set-hooks compile install

# == REFRESH INSTALLATION
refresh-utils-compile:
	# install extra dev group
	uv pip compile pyproject.toml --extra dev -o requirements.txt --cache-dir .uv_cache

reinstall-utils-install: refresh-codeartifact-token
	uv pip sync requirements.txt --cache-dir .uv_cache

local: refresh-utils-compile reinstall-utils-install

# === CODE VALIDATION

format:
	. $(VENV_BIN)/activate && ruff format $(SRC_DIR)

lint:
	. $(VENV_BIN)/activate && ruff check $(SRC_DIR) --fix
	. $(VENV_BIN)/activate && mypy --ignore-missing-imports --install-types --non-interactive --package $(SRC_DIR)

test:
	. $(VENV_BIN)/activate && pytest --verbose --color=yes --cov=$(SRC_DIR) -n auto

all-validation: format lint test
