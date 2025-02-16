check-docker: ## Verify Docker installation and configuration
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "❌ Docker is not installed! Please install it first."; \
		exit 1; \
	elif ! command -v docker-compose >/dev/null 2>&1; then \
		echo "❌ Docker Compose is not installed! Please install it first."; \
		exit 1; \
	else \
		echo "✅ Docker and Docker Compose are installed"; \
	fi

build: ## Create application containers
	@echo "🔨 Building application containers..."
	docker-compose build

rebuild: ## Rebuild containers with fresh configuration
	@echo "🔨 Performing complete rebuild..."
	docker-compose down --volumes --remove-orphans
	docker-compose build --no-cache
	docker-compose up -d

start: ## Launch application services
	@echo "🚀 Starting application services..."
	docker-compose up -d

stop: ## Stop all running services
	@echo "🛑 Stopping application services..."
	docker-compose down

restart: ## Restart all application services
	@echo "🔄 Restarting services..."
	docker-compose down && docker-compose up -d

logs: ## Display container logs
	@echo "📜 Showing application logs..."
	docker-compose logs -f

clean: ## Remove all containers and volumes
	@echo "🧹 Cleaning up resources..."
	docker-compose down --volumes --remove-orphans

run-dev: ## Start development server with live reload
	@echo "🚀 Starting development server..."
	docker-compose up -d redis
	docker-compose up -d app
	docker-compose exec -it lemone-api uv run fastapi dev --port 8687
