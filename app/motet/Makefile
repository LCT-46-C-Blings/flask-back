GOCMD=go
GOBUILD=$(GOCMD) build

BINARY_NAME=motet
BUILD_DIR=./build
MAIN_PACKAGE=.

# Default target
.PHONY:
all: build

# Build for current platform
build:
	@mkdir -p $(BUILD_DIR)
	$(GOBUILD) -o $(BUILD_DIR)/$(BINARY_NAME) $(MAIN_PACKAGE)

# Clean build files
clean:
	rm -rf ./build

.PHONY: all build clean