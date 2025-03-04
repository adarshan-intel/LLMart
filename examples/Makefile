# This Makefile is used to run various example projects and clean up temporary files.

all: run

run: run_basic run_autogcg run_fact_checking run_llmguard run_random_stinrgs run_unlearning

run_autogcg:
	make -C autogcg
	make -C autogcg clean

run_basic:
	make -C basic
	make -C basic clean

run_fact_checking:
	make -C fact_checking
	make -C fact_checking clean

run_llmguard:
	make -C llmguard
	make -C llmguard clean

run_random_stinrgs:
	make -C random_stinrgs
	make -C random_stinrgs clean

run_unlearning:
	make -C unlearning
	make -C unlearning clean

clean:
	rm -rf __pycache__ $(VENV_DIR)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Run all examples"
	@echo "  run        - Run all examples"
	@echo "  run_autogcg- Run autogcg examples"
	@echo "  run_basic  - Run basic examples"
	@echo "  run_fact_checking- Run fact_checking examples"
	@echo "  run_llmguard- Run llmguard examples"
	@echo "  run_random_stinrgs- Run random_stinrgs examples"
	@echo "  run_unlearning- Run unlearning examples"
	@echo "  clean      - Remove temporary files and virtual environment"
	@echo "  help       - Display this help message"

.PHONY: all run run_autogcg run_basic run_fact_checking run_llmguard run_random_stinrgs run_unlearning clean help