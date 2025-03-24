clear_outputs:
	for k in $$(find . -name "*.ipynb" -not -path "*/.*"); do echo "Clearing outputs of $${k}"; jupyter nbconvert --clear-output --inplace $${k}; done
