# This is command "launcher". Used to launch training, model, tests, etc... #

#################### VARIABLES ###################
CURRENT_DIR := $(CURDIR)

#################### PACKAGE ACTIONS ###################

reinstall_package:
	@pip uninstall -y neet || :
	@pip install -e .

run_preprocess:
	python -c 'from neet.interface.main import preprocess; preprocess(source_type="train"); preprocess(source_type="val")'

run_train:
	python -c 'from neet.interface.main import train; train()'

run_pred:
	python -c 'from neet.interface.main import pred; pred()'

run_evaluate:
	python -c 'from neet.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_model: run_all

#################### STREAMLIT ###################

# Streamlit settings
export STREAMLIT_THEME_BASE = light
export STREAMLIT_THEME_PRIMARY_COLOR = 552D62
export STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR = F4F0F6
export STREAMLIT_BROWSER_GATHER_USAGE_STATS = False
export STREAMLIT_CLIENT_TOOLBAR_MODE = minimal

# Run Streamlit
run_streamlit:
	@streamlit run ./neet/streamlit_api/Home.py

#################### TESTS ######################

test:
	@pytest -v tests

#################### DOCKER ######################

docker_show_images:
	@docker images

docker_show_containers:
	@docker ps --all 	

docker_build:
	@docker build -t neet-image .

docker_run_interactive:
	@docker run -it neet-image sh

docker_run:
	@docker run -p 8501:8501 --env-file .env --name neet-container neet-image

docker_remove:
	@docker rm neet-container && docker rmi neet-image

docker_clean:
	@docker image prune	


#################### DATA SOURCES ACTIONS ###################

create_folders:
	mkdir -p /prod/data_dashboard
	mkdir -p /prod/data_develop
	mkdir -p /prod/data_develop/raw
	mkdir -p /prod/data_develop/intermediate
	mkdir -p /prod/data_develop/final