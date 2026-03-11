.PHONY: build run clean check all release

BINARY_NAME=example
BUILD_DIR=bin
SOURCE_DIR=examples/basic

build:
	go build -o $(BUILD_DIR)/$(BINARY_NAME) $(SOURCE_DIR)/main.go

run: build
	$(BUILD_DIR)/$(BINARY_NAME)

clean:
	rm -rf $(BUILD_DIR)

check:
	go vet ./...
	go build ./...

all: clean build

release: check
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make release VERSION=v1.2.3"; \
		exit 1; \
	fi
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: working tree is dirty, commit or stash changes first"; \
		exit 1; \
	fi
	git tag $(VERSION)
	git push origin $(VERSION)
