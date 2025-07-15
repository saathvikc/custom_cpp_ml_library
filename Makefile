# Makefile for Custom C++ ML Library
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
TARGET = ml_library
SRCDIR = .
SOURCES = main.cpp regression/linear_regression.cpp regression/logistic_regression.cpp regression/polynomial_regression.cpp classification/knn.cpp utils/utils.cpp utils/log_manager.cpp tests/linear_regression_tests.cpp tests/logistic_regression_tests.cpp tests/polynomial_regression_tests.cpp tests/knn_tests.cpp

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Debug build with debugging symbols
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Release build with optimizations
release: CXXFLAGS += -O3 -DNDEBUG
release: $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Clean everything including logs
clean-all: clean clean-logs

# Run the program
run: setup $(TARGET)
	./$(TARGET)

# Run only linear regression tests (quick demo + tests)
run-linear: setup $(TARGET)
	@echo "Running Linear Regression Demo and Tests..."
	./$(TARGET) | head -50
	@echo "Check logs/ directory for detailed training logs"

# Run only logistic regression tests (generate data and test)
run-logistic: setup $(TARGET)
	@echo "Running Logistic Regression Demo and Tests..."
	./$(TARGET) | tail -50
	@echo "Check logs/ directory for detailed training logs"

# Create necessary directories
setup:
	mkdir -p models data logs

# View latest training logs
logs:
	@echo "Recent log files:"
	@ls -la logs/ 2>/dev/null || echo "No logs directory found. Run 'make setup' first."

# View specific log file
view-log:
	@read -p "Enter log filename (without .log extension): " logname; \
	if [ -f "logs/$$logname.log" ]; then \
		echo "=== Contents of logs/$$logname.log ==="; \
		cat "logs/$$logname.log"; \
	else \
		echo "Log file logs/$$logname.log not found."; \
		echo "Available logs:"; \
		ls logs/ 2>/dev/null || echo "No logs found."; \
	fi

# Clean logs
clean-logs:
	rm -rf logs/*.log

# Install (copy to /usr/local/bin)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all         - Build the main executable (default)"
	@echo "  debug       - Build with debug symbols"
	@echo "  release     - Build optimized release version"
	@echo "  clean       - Remove build artifacts"
	@echo "  clean-all   - Remove build artifacts and logs"
	@echo "  run         - Build and run the full demo (linear + logistic)"
	@echo "  run-linear  - Run linear regression demo only"
	@echo "  run-logistic- Run logistic regression demo only"
	@echo "  setup       - Create necessary directories"
	@echo "  logs        - List available log files"
	@echo "  view-log    - View contents of a specific log file"
	@echo "  clean-logs  - Remove all log files"
	@echo "  install     - Install to /usr/local/bin"
	@echo "  help        - Show this help message"

.PHONY: all debug release clean clean-all run run-linear run-logistic setup logs view-log clean-logs install uninstall help
