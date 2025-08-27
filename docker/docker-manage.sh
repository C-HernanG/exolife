#!/bin/bash
# ExoLife Docker Management Script
# Provides convenient commands for Docker operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project settings
PROJECT_NAME="exolife"
IMAGE_NAME="exolife"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to build Docker image
build() {
    local target="${1:-production}"
    log_info "Building Docker image for $target..."
    
    case $target in
        "dev"|"development")
            docker build --target builder -t ${IMAGE_NAME}:dev .
            ;;
        "prod"|"production")
            docker build -t ${IMAGE_NAME}:latest -t ${IMAGE_NAME}:prod .
            ;;
        *)
            log_error "Invalid build target. Use 'dev' or 'prod'"
            exit 1
            ;;
    esac
    
    log_success "Docker image built successfully"
}

# Function to run development environment
dev() {
    log_info "Starting development environment..."
    docker-compose -f docker-compose.dev.yml up -d
    log_success "Development environment started"
    log_info "Jupyter Lab available at: http://localhost:8888"
    log_info "To attach to development container: docker exec -it exolife-dev bash"
}

# Function to run production environment
prod() {
    log_info "Starting production environment..."
    docker-compose -f docker-compose.prod.yml up -d
    log_success "Production environment started"
}

# Function to stop all services
stop() {
    log_info "Stopping all ExoLife services..."
    docker-compose -f docker-compose.yml down 2>/dev/null || true
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    log_success "All services stopped"
}

# Function to clean up Docker resources
clean() {
    log_info "Cleaning up Docker resources..."
    
    # Stop all containers
    stop
    
    # Remove containers
    docker container prune -f
    
    # Remove images
    docker rmi ${IMAGE_NAME}:latest ${IMAGE_NAME}:dev ${IMAGE_NAME}:prod 2>/dev/null || true
    
    # Remove unused volumes (with confirmation)
    read -p "Remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    log_success "Cleanup completed"
}

# Function to show logs
logs() {
    local service="${1:-exolife}"
    local environment="${2:-dev}"
    
    case $environment in
        "dev"|"development")
            docker-compose -f docker-compose.dev.yml logs -f $service
            ;;
        "prod"|"production")
            docker-compose -f docker-compose.prod.yml logs -f $service
            ;;
        *)
            docker-compose logs -f $service
            ;;
    esac
}

# Function to run ExoLife CLI commands in container
cli() {
    if [ $# -eq 0 ]; then
        log_error "Please provide a command to run"
        exit 1
    fi
    
    local cmd="$*"
    log_info "Running: exolife $cmd"
    
    docker run --rm -it \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/config:/app/config:ro" \
        -v "$(pwd)/results:/app/results" \
        ${IMAGE_NAME}:latest \
        exolife $cmd
}

# Function to run interactive bash session
shell() {
    local environment="${1:-dev}"
    
    case $environment in
        "dev"|"development")
            docker exec -it exolife-jupyter-dev bash
            ;;
        "prod"|"production")
            docker exec -it exolife-prod bash
            ;;
        *)
            docker run --rm -it \
                -v "$(pwd)/data:/app/data" \
                -v "$(pwd)/config:/app/config:ro" \
                -v "$(pwd)/results:/app/results" \
                ${IMAGE_NAME}:latest \
                bash
            ;;
    esac
}

# Function to run Jupyter Lab
jupyter() {
    log_info "Starting Jupyter Lab..."
    docker run --rm -it \
        -p 8888:8888 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/notebooks:/app/notebooks" \
        -v "$(pwd)/config:/app/config:ro" \
        -v "$(pwd)/results:/app/results" \
        ${IMAGE_NAME}:latest \
        jupyter lab \
        --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --NotebookApp.token='' --NotebookApp.password=''
}

# Function to show status
status() {
    log_info "Docker containers status:"
    docker ps -a --filter "name=exolife" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    log_info "Docker images:"
    docker images --filter "reference=${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo ""
    log_info "Docker volumes:"
    docker volume ls --filter "name=exolife" --format "table {{.Name}}\t{{.Driver}}\t{{.Size}}"
}

# Function to show help
help() {
    cat << EOF
ExoLife Docker Management Script

Usage: $0 <command> [options]

Commands:
    build [dev|prod]     Build Docker image (default: prod)
    dev                  Start development environment
    prod                 Start production environment
    stop                 Stop all services
    clean                Clean up Docker resources
    logs [service] [env] Show logs (default: exolife, dev)
    cli <command>        Run ExoLife CLI command in container
    shell [env]          Open interactive shell (default: dev)
    jupyter              Start Jupyter Lab
    status               Show Docker status
    help                 Show this help message

Examples:
    $0 build dev         Build development image
    $0 dev               Start development environment
    $0 cli dag run config/dags/dagspec.yaml
    $0 shell prod        Open shell in production container
    $0 logs jupyter dev  Show Jupyter logs in dev environment

EOF
}

# Main script logic
main() {
    check_docker
    
    case "${1:-help}" in
        "build")
            build "${2:-prod}"
            ;;
        "dev"|"development")
            dev
            ;;
        "prod"|"production")
            prod
            ;;
        "stop")
            stop
            ;;
        "clean")
            clean
            ;;
        "logs")
            logs "${2:-exolife}" "${3:-dev}"
            ;;
        "cli")
            shift
            cli "$@"
            ;;
        "shell")
            shell "${2:-dev}"
            ;;
        "jupyter")
            jupyter
            ;;
        "status")
            status
            ;;
        "help"|"-h"|"--help")
            help
            ;;
        *)
            log_error "Unknown command: $1"
            help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
