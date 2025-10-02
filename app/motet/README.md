# Motet - Medical Data Simulator
A Go-based application for simulating and sending medical data from CSV files to a remote server. 

## Overview
Motet reads time-series data from CSV files and sends it to a specified HTTP endpoint with precise timing, simulating real-time data streams from medical sensors. It supports multiple data types (e.g., BPM and uterine activity) and can run in loop mode for continuous testing.

## Installation

### Prerequisites
    Go 1.25 or later

### Building
    # Clone the repository
    git clone <repository-url>
    cd motet

    # Build the application
    make build
    # or directly
    go build -o build/motet .
The binary will be created in the build/ directory.

### Usage
    ./build/motet -bpm <bpm_csv_file> -uterus <uterus_csv_file> -url <target_url> [-loop]

### Command Line Options
- -bpm: Path to CSV file containing heart rate (BPM) data
- -uterus: Path to CSV file containing uterine activity data
- -url: Target URL for sending data (default: localhost:8080)
- -loop: Enable continuous looping of data (default: false)
- -help: Show help message
### CSV Format
CSV files should follow this format:

    time,value
    0.0,72.5
    1.5,73.2
    3.0,72.8
    ...
- Header: First line is required (will be skipped)
- Time: Floating-point number representing seconds from start
- Value: Floating-point measurement value

### API Endpoints
Data is sent via HTTP POST requests to:

BPM data: http://\<url>/bpm?value=\<measurement>

Uterus data: http://\<url>/uterus?value=\<measurement>