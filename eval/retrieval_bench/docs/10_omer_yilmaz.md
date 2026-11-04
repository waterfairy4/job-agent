# Ömer Yılmaz (fictional candidate) — project notes

## Real-time data pipeline
- Built a streaming pipeline that ingests 2 million events per minute from Kafka into ClickHouse for product analytics.
- Airflow schedules daily batch jobs that rebuild aggregate tables; data quality checks stop the pipeline when row counts drop by more than 5%.
- Schema changes are versioned with a schema registry so producers cannot break consumers.

## Network monitoring with eBPF
- Wrote eBPF programs that record TCP retransmissions and round-trip times per connection inside Kubernetes pods, with less than 1% CPU overhead.
- Found that a misconfigured MTU on one node caused most retransmissions; fixing it cut tail latency by 30%.
- Dashboards in Grafana show retransmission rate per service and alert when it exceeds 2%.

## Log anomaly detection
- Parsed 30 GB of service logs per day into templates with Drain, then flagged unusual template frequencies with an isolation forest.
- Alerts are grouped by service and deployment version, which reduced alert noise for the on-call team by about 60%.
- Evaluated on 14 past incidents: 11 were detected before the first customer complaint.

## University project: congestion control comparison
- Compared TCP Cubic and BBR in a Mininet testbed with different buffer sizes; BBR kept lower queueing delay but was unfair to Cubic flows in shallow buffers.

## Skills
- Python, Go, C (eBPF); Kafka, ClickHouse, Airflow; Linux networking, TCP.
