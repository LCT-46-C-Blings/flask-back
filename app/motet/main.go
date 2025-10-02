package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// DataPoint represents a single row in the CSV
type DataPoint struct {
	Time  float64
	Value float64
}

// ParseCSV parses the CSV file and returns a slice of DataPoints
func ParseCSV(filename string) ([]DataPoint, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ','
	reader.Comment = '#'
	reader.FieldsPerRecord = -1 // Allow variable number of fields

	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	var dataPoints []DataPoint

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read record: %w", err)
		}

		// Skip empty records
		if len(record) < 2 {
			continue
		}

		// Parse time
		time, err := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse time '%s': %w", record[0], err)
		}

		// Parse value
		value, err := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse value '%s': %w", record[1], err)
		}

		dataPoints = append(dataPoints, DataPoint{
			Time:  time,
			Value: value,
		})
	}

	return dataPoints, nil
}

func sendData(points []DataPoint, url string) {
	start := time.Now()
	client := &http.Client{}

	for _, point := range points {
		targetTime := start.Add(time.Duration(point.Time * float64(time.Second)))
		time.Sleep(time.Until(targetTime))

		url := fmt.Sprintf("%s?value=%f", url, point.Value)
		resp, err := client.Post(url, "application/json", nil)

		if err != nil {
			fmt.Printf("❌ Error at %.3fs: %v\n", point.Time, err)
		} else {
			fmt.Printf("✅ %.3fs: value=%.6f (status=%d)\n",
				point.Time, point.Value, resp.StatusCode)
			resp.Body.Close()
		}
	}
}

func main() {
	flagBpm := flag.String("bpm", "", "path to bpm csv data file")
	flagUterus := flag.String("uterus", "", "path to uterus csv data file")
	flagUrl := flag.String("url", "localhost:8080", "defines the url to which the data will be sent")
	flagLoop := flag.Bool("loop", false, "loop sending data")
	flagHelp := flag.Bool("help", false, "shows this message")
	flag.Parse()

	if *flagHelp {
		flag.Usage()
		return
	}

	bpms, err := ParseCSV(*flagBpm)
	if err != nil {
		log.Fatalf("Failed to parse bpms csv: %v", err)
	}
	fmt.Printf("📁 Loaded %d data points from bpm CSV\n", len(bpms))

	uterus, err := ParseCSV(*flagUterus)
	if err != nil {
		log.Fatalf("Failed to parse uterus csv: %v", err)
	}
	fmt.Printf("📁 Loaded %d data points from uterus CSV\n", len(uterus))

	fmt.Printf("Start sending data...")

	for {
		wg := sync.WaitGroup{}

		wg.Go(func() {
			fmt.Printf("Start sending bpms...")
			sendData(bpms, "http://"+*flagUrl+"/bpm")
			fmt.Println("✅ All bpms requests completed!")
		})

		wg.Go(func() {
			fmt.Printf("Start sending uterus...")
			sendData(uterus, "http://"+*flagUrl+"/uterus")
			fmt.Println("✅ All uterus requests completed!")
		})

		wg.Wait()

		if !*flagLoop {
			break
		}
	}

	fmt.Println("✅ All requests completed!")
}
