.PHONY: all features lattice model evaluate smoke clean

all: features lattice model evaluate

features:
	python src/features.py

lattice:
	python src/lattice_pricer.py

model:
	python src/xgb_model.py

evaluate:
	python src/evaluation.py

smoke:
	python binomial.py

clean:
	rm -f data/nvda_with_lattice.csv data/nvda_test_predictions.csv models/*.json figures/*.png
