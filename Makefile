.PHONY: help build-api build-dashboard build-all push-api push-dashboard push-all compose-up compose-down compose-logs compose-ps restart shell-api shell-dashboard clean-images clean-volumes

# Docker-centric workflow for Chatbot Analytics
DOCKER ?= docker
DOCKER_COMPOSE ?= docker compose
COMPOSE_FILE ?= docker-compose.yml
REGISTRY ?=
TAG ?= latest

IMAGE_PREFIX := $(if $(strip $(REGISTRY)),$(REGISTRY)/,)
API_IMAGE := $(IMAGE_PREFIX)chatbot-analytics-api
DASHBOARD_IMAGE := $(IMAGE_PREFIX)chatbot-analytics-dashboard
API_IMAGE_TAGGED := $(API_IMAGE):$(TAG)
DASHBOARD_IMAGE_TAGGED := $(DASHBOARD_IMAGE):$(TAG)

help:
	@echo "Chatbot Analytics Docker Commands"
	@echo "---------------------------------"
	@echo "Images:"
	@echo "  make build-api          Build the API image ($(API_IMAGE_TAGGED))"
	@echo "  make build-dashboard    Build the dashboard image ($(DASHBOARD_IMAGE_TAGGED))"
	@echo "  make build-all          Build both project images"
	@echo "  make push-api           Push the API image"
	@echo "  make push-dashboard     Push the dashboard image"
	@echo "  make push-all           Push both project images"
	@echo ""
	@echo "Compose:"
	@echo "  make compose-up         Start services with docker compose"
	@echo "  make compose-down       Stop services and remove containers"
	@echo "  make restart            Restart services (down + up)"
	@echo "  make compose-logs       Tail compose logs (Ctrl+C to exit)"
	@echo "  make compose-ps         Show running compose services"
	@echo ""
	@echo "Utilities:"
	@echo "  make shell-api          Run an interactive shell in the API image"
	@echo "  make shell-dashboard    Run an interactive shell in the dashboard image"
	@echo "  make clean-images       Remove local images for the project"
	@echo "  make clean-volumes      Remove compose-managed volumes"
	@echo ""
	@echo "Set REGISTRY=<registry> and TAG=<tag> to customize image destinations."

build-api:
	$(DOCKER) build --pull --rm -f Dockerfile.api -t $(API_IMAGE_TAGGED) .

build-dashboard:
	$(DOCKER) build --pull --rm -f Dockerfile.dashboard -t $(DASHBOARD_IMAGE_TAGGED) .

build-all: build-api build-dashboard

push-api:
	$(DOCKER) push $(API_IMAGE_TAGGED)

push-dashboard:
	$(DOCKER) push $(DASHBOARD_IMAGE_TAGGED)

push-all: push-api push-dashboard

compose-up:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d

compose-down:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down

restart: compose-down compose-up

compose-logs:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f

compose-ps:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps

shell-api:
	$(DOCKER) run --rm -it $(API_IMAGE_TAGGED) /bin/bash

shell-dashboard:
	$(DOCKER) run --rm -it $(DASHBOARD_IMAGE_TAGGED) /bin/bash

clean-images:
	-$(DOCKER) image rm -f $(API_IMAGE_TAGGED) $(DASHBOARD_IMAGE_TAGGED)

clean-volumes:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v
