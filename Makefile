.PHONY: install train test lint run docker-build k8s-apply tf-plan clean

install:        ## Install dev dependencies
	pip install -r requirements-dev.txt

train:          ## Train the model -> model/flight_fare_rf.pkl
	python -m scripts.train

test:           ## Run the test suite
	pytest

lint:           ## Lint the code
	flake8 app scripts tests wsgi.py

run:            ## Run locally with gunicorn on :8080
	gunicorn --bind 0.0.0.0:8080 --workers 2 --threads 4 wsgi:app

docker-build:   ## Build the container image
	docker build -t flight-fare:local .

k8s-apply:      ## Apply manifests to the current kube-context
	kubectl apply -k infra/k8s

tf-plan:        ## Terraform plan for the EKS/ECR stack
	cd infra/terraform && terraform init && terraform plan

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; rm -rf .pytest_cache
