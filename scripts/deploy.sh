#!/bin/bash
# Deployment script for Options Pricing Engine

set -e

echo "========================================="
echo "Options Pricing Engine - Deployment"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Navigate to deployment directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../deployment"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from .env.example${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please update .env with your configuration before proceeding${NC}"
    exit 1
fi

# Load environment variables
source .env

echo -e "${GREEN}[1/6] Stopping existing containers...${NC}"
docker-compose down

echo -e "${GREEN}[2/6] Building Docker images...${NC}"
docker-compose build --no-cache

echo -e "${GREEN}[3/6] Starting services...${NC}"
docker-compose up -d

echo -e "${GREEN}[4/6] Waiting for services to be healthy...${NC}"
sleep 10

# Check if services are running
echo -e "${GREEN}[5/6] Checking service health...${NC}"

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U ${POSTGRES_USER} &> /dev/null; then
    echo -e "${GREEN}✓ PostgreSQL is running${NC}"
else
    echo -e "${RED}✗ PostgreSQL is not responding${NC}"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${RED}✗ Redis is not responding${NC}"
fi

# Check Pricing Service
if curl -s http://localhost:5000/health &> /dev/null; then
    echo -e "${GREEN}✓ Pricing Service is running${NC}"
else
    echo -e "${YELLOW}⚠ Pricing Service is not yet responding (may still be starting)${NC}"
fi

# Check API Gateway
if curl -s http://localhost:8080/health &> /dev/null; then
    echo -e "${GREEN}✓ API Gateway is running${NC}"
else
    echo -e "${YELLOW}⚠ API Gateway is not yet responding (may still be starting)${NC}"
fi

echo -e "${GREEN}[6/6] Running database migrations...${NC}"
docker-compose exec -T postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -f /docker-entrypoint-initdb.d/001_initial_schema.sql

echo ""
echo "========================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "========================================="
echo ""
echo "Service URLs:"
echo "  API Gateway:  http://localhost:8080"
echo "  Pricing Service: http://localhost:5000"
echo "  Prometheus:   http://localhost:9090"
echo "  Grafana:      http://localhost:3000"
echo "  PostgreSQL:   localhost:5432"
echo "  Redis:        localhost:6379"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f [service-name]"
echo ""
echo "To stop all services:"
echo "  docker-compose down"
echo ""
echo "To test the API:"
echo "  curl http://localhost:8080/health"
echo ""
