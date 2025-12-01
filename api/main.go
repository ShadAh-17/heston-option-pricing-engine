package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
)

var (
	// Prometheus metrics
	requestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "api_requests_total",
			Help: "Total number of API requests",
		},
		[]string{"endpoint", "method", "status"},
	)
	requestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "api_request_duration_seconds",
			Help:    "API request duration in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0},
		},
		[]string{"endpoint"},
	)
)

func init() {
	prometheus.MustRegister(requestCounter)
	prometheus.MustRegister(requestDuration)
}

// Application holds all dependencies
type Application struct {
	redis  *redis.Client
	logger *log.Logger
}

// PriceRequest represents a pricing request
type PriceRequest struct {
	Model      string  `json:"model"`       // "heston", "merton", "heston-merton"
	Spot       float64 `json:"spot"`        // S0
	Strike     float64 `json:"strike"`      // K
	Rate       float64 `json:"rate"`        // r
	Dividend   float64 `json:"dividend"`    // q
	Maturity   float64 `json:"maturity"`    // T
	OptionType string  `json:"option_type"` // "call" or "put"
	Method     string  `json:"method"`      // "fft" or "mc"
	// Heston parameters
	V0    float64 `json:"v0,omitempty"`
	Kappa float64 `json:"kappa,omitempty"`
	Theta float64 `json:"theta,omitempty"`
	Sigma float64 `json:"sigma,omitempty"`
	Rho   float64 `json:"rho,omitempty"`
}

// PriceResponse represents a pricing response
type PriceResponse struct {
	Price     float64 `json:"price"`
	Method    string  `json:"method"`
	Timestamp string  `json:"timestamp"`
	Duration  float64 `json:"duration_ms"`
	Cached    bool    `json:"cached"`
}

// GreeksRequest represents a Greeks calculation request
type GreeksRequest struct {
	PriceRequest
}

// GreeksResponse represents Greeks calculation response
type GreeksResponse struct {
	Delta     float64 `json:"delta"`
	Gamma     float64 `json:"gamma"`
	Vega      float64 `json:"vega"`
	Theta     float64 `json:"theta"`
	Price     float64 `json:"price"`
	Timestamp string  `json:"timestamp"`
	Duration  float64 `json:"duration_ms"`
}

// CalibrateRequest represents a calibration request
type CalibrateRequest struct {
	Spot       float64            `json:"spot"`
	Rate       float64            `json:"rate"`
	Dividend   float64            `json:"dividend"`
	Model      string             `json:"model"`  // "heston", "merton", "heston-merton"
	Method     string             `json:"method"` // "global" or "local"
	MarketData []MarketOptionData `json:"market_data"`
}

// MarketOptionData represents a single market option
type MarketOptionData struct {
	Strike     float64 `json:"strike"`
	Maturity   float64 `json:"maturity"`
	Price      float64 `json:"price"`
	OptionType string  `json:"option_type"`
}

// CalibrateResponse represents calibration response
type CalibrateResponse struct {
	Parameters map[string]float64 `json:"parameters"`
	RMSE       float64            `json:"rmse"`
	Success    bool               `json:"success"`
	Duration   float64            `json:"duration_ms"`
	Timestamp  string             `json:"timestamp"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
	Code    int    `json:"code"`
}

func main() {
	// Initialize logger
	logger := log.New(os.Stdout, "[OPTIONS-API] ", log.LstdFlags|log.Lshortfile)

	// Initialize Redis
	redisClient := redis.NewClient(&redis.Options{
		Addr:     getEnv("REDIS_URL", "localhost:6379"),
		Password: getEnv("REDIS_PASSWORD", ""),
		DB:       0,
	})

	// Test Redis connection
	ctx := context.Background()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		logger.Printf("WARNING: Redis connection failed: %v (running without cache)", err)
	} else {
		logger.Println("Connected to Redis successfully")
	}

	app := &Application{
		redis:  redisClient,
		logger: logger,
	}

	// Create router
	router := mux.NewRouter()

	// API routes
	router.HandleFunc("/health", app.healthHandler).Methods("GET")
	router.HandleFunc("/api/v1/price", app.metricsMiddleware(app.priceHandler)).Methods("POST")
	router.HandleFunc("/api/v1/greeks", app.metricsMiddleware(app.greeksHandler)).Methods("POST")
	router.HandleFunc("/api/v1/calibrate", app.metricsMiddleware(app.calibrateHandler)).Methods("POST")
	router.HandleFunc("/api/v1/surface", app.metricsMiddleware(app.surfaceHandler)).Methods("POST")

	// Metrics endpoint
	router.Handle("/metrics", promhttp.Handler())

	// CORS middleware
	router.Use(corsMiddleware)

	// Server configuration
	port := getEnv("PORT", "8080")
	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	go func() {
		logger.Printf("Starting server on port %s", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf("Server error: %v", err)
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt)
	<-quit

	logger.Println("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Printf("Server forced to shutdown: %v", err)
	}

	logger.Println("Server stopped")
}

// Health check handler
func (app *Application) healthHandler(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Format(time.RFC3339),
		"redis":     app.checkRedis(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// Price handler
func (app *Application) priceHandler(w http.ResponseWriter, r *http.Request) {
	var req PriceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		app.sendError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate request
	if err := app.validatePriceRequest(&req); err != nil {
		app.sendError(w, err.Error(), http.StatusBadRequest)
		return
	}

	start := time.Now()

	// Check cache first
	cacheKey := app.generateCacheKey(&req)
	cachedPrice, found := app.getFromCache(cacheKey)

	var price float64
	var cached bool

	if found {
		price = cachedPrice
		cached = true
		app.logger.Println("Cache hit for pricing request")
	} else {
		// Call Python pricing service
		var err error
		price, err = app.callPricingService(&req)
		if err != nil {
			app.sendError(w, fmt.Sprintf("Pricing error: %v", err), http.StatusInternalServerError)
			return
		}

		// Cache the result (TTL: 5 minutes)
		app.setCache(cacheKey, price, 5*time.Minute)
		cached = false
	}

	duration := time.Since(start).Seconds() * 1000

	response := PriceResponse{
		Price:     price,
		Method:    req.Method,
		Timestamp: time.Now().Format(time.RFC3339),
		Duration:  duration,
		Cached:    cached,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Greeks handler
func (app *Application) greeksHandler(w http.ResponseWriter, r *http.Request) {
	var req GreeksRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		app.sendError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	start := time.Now()

	// Call Python service for Greeks calculation
	greeks, err := app.callGreeksService(&req)
	if err != nil {
		app.sendError(w, fmt.Sprintf("Greeks calculation error: %v", err), http.StatusInternalServerError)
		return
	}

	duration := time.Since(start).Seconds() * 1000
	greeks.Duration = duration
	greeks.Timestamp = time.Now().Format(time.RFC3339)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(greeks)
}

// Calibrate handler
func (app *Application) calibrateHandler(w http.ResponseWriter, r *http.Request) {
	var req CalibrateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		app.sendError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if len(req.MarketData) == 0 {
		app.sendError(w, "Market data is required", http.StatusBadRequest)
		return
	}

	start := time.Now()

	// Call Python service for calibration
	result, err := app.callCalibrationService(&req)
	if err != nil {
		app.sendError(w, fmt.Sprintf("Calibration error: %v", err), http.StatusInternalServerError)
		return
	}

	duration := time.Since(start).Seconds() * 1000
	result.Duration = duration
	result.Timestamp = time.Now().Format(time.RFC3339)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// Surface handler (prices multiple strikes/maturities at once)
func (app *Application) surfaceHandler(w http.ResponseWriter, r *http.Request) {
	// TODO: Implement surface pricing
	app.sendError(w, "Surface endpoint not yet implemented", http.StatusNotImplemented)
}

// Helper functions

func (app *Application) validatePriceRequest(req *PriceRequest) error {
	if req.Spot <= 0 {
		return fmt.Errorf("spot price must be positive")
	}
	if req.Strike <= 0 {
		return fmt.Errorf("strike price must be positive")
	}
	if req.Maturity <= 0 {
		return fmt.Errorf("maturity must be positive")
	}
	if req.OptionType != "call" && req.OptionType != "put" {
		return fmt.Errorf("option_type must be 'call' or 'put'")
	}
	if req.Method != "fft" && req.Method != "mc" {
		req.Method = "fft" // Default to FFT
	}
	return nil
}

func (app *Application) generateCacheKey(req *PriceRequest) string {
	return fmt.Sprintf("price:%s:%s:%.2f:%.2f:%.4f:%.4f:%.4f",
		req.Model, req.OptionType, req.Spot, req.Strike, req.Rate, req.Dividend, req.Maturity)
}

func (app *Application) getFromCache(key string) (float64, bool) {
	ctx := context.Background()
	val, err := app.redis.Get(ctx, key).Float64()
	if err != nil {
		return 0, false
	}
	return val, true
}

func (app *Application) setCache(key string, value float64, ttl time.Duration) {
	ctx := context.Background()
	app.redis.Set(ctx, key, value, ttl)
}

func (app *Application) checkRedis() string {
	ctx := context.Background()
	if err := app.redis.Ping(ctx).Err(); err != nil {
		return "unhealthy"
	}
	return "healthy"
}

func (app *Application) sendError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)

	err := ErrorResponse{
		Error:   http.StatusText(code),
		Message: message,
		Code:    code,
	}
	json.NewEncoder(w).Encode(err)
	app.logger.Printf("Error %d: %s", code, message)
}

// Middleware

func (app *Application) metricsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Create custom response writer to capture status
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next(rw, r)

		duration := time.Since(start).Seconds()

		// Record metrics
		requestCounter.WithLabelValues(r.URL.Path, r.Method, fmt.Sprintf("%d", rw.statusCode)).Inc()
		requestDuration.WithLabelValues(r.URL.Path).Observe(duration)
	}
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// Custom response writer to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Utility functions

func getEnv(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}

// Python service communication (implement these based on your Python service setup)

func (app *Application) callPricingService(req *PriceRequest) (float64, error) {
	// TODO: Call Python pricing service via HTTP or gRPC
	// For now, return mock data
	app.logger.Println("Calling Python pricing service...")
	return 8.5234, nil
}

func (app *Application) callGreeksService(req *GreeksRequest) (*GreeksResponse, error) {
	// TODO: Call Python Greeks service
	app.logger.Println("Calling Python Greeks service...")
	return &GreeksResponse{
		Delta: 0.5234,
		Gamma: 0.0123,
		Vega:  15.234,
		Theta: -0.0234,
		Price: 8.5234,
	}, nil
}

func (app *Application) callCalibrationService(req *CalibrateRequest) (*CalibrateResponse, error) {
	// TODO: Call Python calibration service
	app.logger.Println("Calling Python calibration service...")
	return &CalibrateResponse{
		Parameters: map[string]float64{
			"v0":    0.04,
			"kappa": 2.0,
			"theta": 0.04,
			"sigma": 0.5,
			"rho":   -0.7,
		},
		RMSE:    0.0023,
		Success: true,
	}, nil
}
