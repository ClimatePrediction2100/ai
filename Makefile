.PHONY: expr setup

expr:
	@bash expr.sh

setup:
	@pip install -r requirements.txt
	@mkdir results