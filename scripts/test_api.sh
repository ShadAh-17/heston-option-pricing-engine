#!/bin/bash
# Test script for Options Pricing Engine API

set -e

API_URL="${API_URL:-http://localhost:8080}"

echo "========================================="
echo "Testing Options Pricing Engine API"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test health endpoint
echo "Test 1: Health Check"
response=$(curl -s -w "\n%{http_code}" $API_URL/health)
http_code=$(echo "$response" | tail -n 1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" == "200" ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "$body" | jq '.'
else
    echo -e "${RED}✗ Health check failed (HTTP $http_code)${NC}"
    exit 1
fi
echo ""

# Test pricing endpoint
echo "Test 2: Option Pricing (Heston FFT)"
response=$(curl -s -w "\n%{http_code}" -X POST $API_URL/api/v1/price \
  -H "Content-Type: application/json" \
  -d '{
    "spot": 100,
    "strike": 100,
    "rate": 0.05,
    "dividend": 0.02,
    "maturity": 1.0,
    "option_type": "call",
    "v0": 0.04,
    "kappa": 2.0,
    "theta": 0.04,
    "sigma": 0.3,
    "rho": -0.7,
    "method": "fft"
  }')

http_code=$(echo "$response" | tail -n 1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" == "200" ]; then
    echo -e "${GREEN}✓ Pricing test passed${NC}"
    echo "$body" | jq '.'
else
    echo -e "${RED}✗ Pricing test failed (HTTP $http_code)${NC}"
    echo "$body"
    exit 1
fi
echo ""

# Test Greeks endpoint
echo "Test 3: Greeks Calculation"
response=$(curl -s -w "\n%{http_code}" -X POST $API_URL/api/v1/greeks \
  -H "Content-Type: application/json" \
  -d '{
    "spot": 100,
    "strike": 100,
    "rate": 0.05,
    "dividend": 0.02,
    "maturity": 1.0,
    "option_type": "call",
    "v0": 0.04,
    "kappa": 2.0,
    "theta": 0.04,
    "sigma": 0.3,
    "rho": -0.7
  }')

http_code=$(echo "$response" | tail -n 1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" == "200" ]; then
    echo -e "${GREEN}✓ Greeks test passed${NC}"
    echo "$body" | jq '.'
else
    echo -e "${RED}✗ Greeks test failed (HTTP $http_code)${NC}"
    echo "$body"
    exit 1
fi
echo ""

# Test calibration endpoint
echo "Test 4: Model Calibration"
response=$(curl -s -w "\n%{http_code}" -X POST $API_URL/api/v1/calibrate \
  -H "Content-Type: application/json" \
  -d '{
    "spot": 100,
    "rate": 0.05,
    "dividend": 0.02,
    "strikes": [90, 95, 100, 105, 110],
    "maturities": [0.25, 0.5, 1.0],
    "market_prices": [
      [12.5, 8.2, 5.1, 3.0, 1.8],
      [14.8, 10.9, 7.8, 5.5, 3.9],
      [18.2, 14.5, 11.5, 9.0, 7.1]
    ],
    "option_type": "call"
  }')

http_code=$(echo "$response" | tail -n 1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" == "200" ]; then
    echo -e "${GREEN}✓ Calibration test passed${NC}"
    echo "$body" | jq '.parameters, .metrics'
else
    echo -e "${RED}✗ Calibration test failed (HTTP $http_code)${NC}"
    echo "$body"
    exit 1
fi
echo ""

# Test volatility surface endpoint
echo "Test 5: Volatility Surface"
response=$(curl -s -w "\n%{http_code}" -X POST $API_URL/api/v1/surface \
  -H "Content-Type: application/json" \
  -d '{
    "spot": 100,
    "rate": 0.05,
    "dividend": 0.02,
    "strikes": [90, 95, 100, 105, 110],
    "maturities": [0.25, 0.5, 1.0],
    "v0": 0.04,
    "kappa": 2.0,
    "theta": 0.04,
    "sigma": 0.3,
    "rho": -0.7,
    "option_type": "call"
  }')

http_code=$(echo "$response" | tail -n 1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" == "200" ]; then
    echo -e "${GREEN}✓ Surface test passed${NC}"
    echo "$body" | jq '.computation_time_ms'
else
    echo -e "${RED}✗ Surface test failed (HTTP $http_code)${NC}"
    echo "$body"
    exit 1
fi
echo ""

echo "========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo "========================================="
