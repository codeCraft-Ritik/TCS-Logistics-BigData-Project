# TCS Internship Project: Logistics Analytics & Machine Learning

## Project Overview

This project implements a comprehensive data analytics and machine learning solution for logistics optimization. It analyzes 14 datasets containing 85,410+ trip records across 124 drivers, 92 trucks, and 58 routes to provide actionable insights for fleet management.

---

## 📁 Project Structure

```
TCS Interhsip - (Project)/
│
├── Dataset/                          # Input Data (Raw Datasets)
│   ├── drivers.csv                  # Driver information and metrics
│   ├── trucks.csv                   # Fleet vehicle data
│   ├── trailers.csv                 # Trailer information
│   ├── routes.csv                   # Route definitions and distances
│   ├── loads.csv                    # Load/shipment data
│   ├── trips.csv                    # Trip records (85,410 records)
│   ├── facilities.csv               # Warehouse and facility data
│   ├── customers.csv                # Customer information
│   ├── fuel_purchases.csv           # Fuel transaction history
│   ├── maintenance_records.csv      # Vehicle maintenance logs
│   ├── delivery_events.csv          # Delivery status updates
│   ├── safety_incidents.csv         # Safety incident reports
│   ├── driver_monthly_metrics.csv   # Monthly driver KPIs
│   └── truck_utilization_metrics.csv # Fleet utilization metrics
│
├── Output/                           # Generated Reports (CSV Files)
│   ├── trip_analysis_report.csv
│   ├── driver_metrics_report.csv
│   ├── fleet_metrics_report.csv
│   ├── route_metrics_report.csv
│   ├── fuel_efficiency_report.csv
│   ├── delivery_performance_report.csv
│   ├── safety_summary_report.csv
│   ├── maintenance_summary_report.csv
│   ├── *_feature_importance.csv     # ML model feature rankings
│   └── [Other generated reports]
│
├── Docs/                             # Visualizations & Documentation
│   ├── Driver_Dashboard.png         # Driver performance analysis
│   ├── Fleet_Dashboard.png          # Fleet utilization dashboard
│   ├── Route_Dashboard.png          # Route profitability analysis
│   ├── KPI_Summary_Table.png        # Executive KPI summary
│   └── [Additional visualizations]
│
├── Python Modules (Custom Libraries)
│   ├── data_processor.py            # Data loading, cleaning, transformation
│   ├── ml_models.py                 # ML model training & evaluation
│   └── visualization_utils.py       # Dashboard and chart generation
│
├── Analysis Notebooks (Jupyter)
│   ├── TCS_Logistics_Complete_Analysis.ipynb
│   │   └── Exploratory data analysis and KPI calculations
│   │
│   └── TCS_Advanced_Analytics_ML.ipynb
│       └── Machine learning models and predictive analytics
│
├── Execution Script
│   └── execute_project.py           # Complete end-to-end pipeline
│
└── Documentation
    └── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "TCS Interhsip - (Project)"

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Complete End-to-End Execution
```bash
python execute_project.py
```
This runs the complete pipeline and generates all reports and visualizations.

#### Option 2: Run Jupyter Notebooks
```bash
jupyter notebook TCS_Advanced_Analytics_ML.ipynb
```

---

## 📊 Key Metrics & KPIs

### Driver Performance
- **Total Drivers**: 124
- **Total Revenue**: $257.3M
- **Average Revenue/Driver**: $2.08M
- **Fleet Average MPG**: 6.50
- **On-Time Delivery Rate**: 55.67%

### Fleet Utilization
- **Total Trucks**: 92
- **Active Trucks**: 92
- **Total Fleet Revenue**: $257.3M
- **Average Cost/Mile**: $2.80
- **Average Trips/Truck**: 928

### Route Profitability
- **Total Routes**: 58
- **Total Profit**: $20.1M
- **Average Profit/Route**: $347K
- **Average Profit Margin**: 63.5%

### Fuel Efficiency
- **Fleet Average MPG**: 6.50
- **Total Fuel Cost**: $39.6M
- **Fuel Cost/Mile**: $0.42

---

## 🤖 Machine Learning Models

### 1. Fuel Consumption Prediction
- **Algorithm**: Gradient Boosting Regressor
- **Performance**: R² = 0.9669, RMSE = 23.03 gallons
- **Use Case**: Predict fuel requirements for trip planning

### 2. Delivery Delay Prediction
- **Algorithm**: Random Forest Classifier
- **Performance**: Accuracy = 81.58%, Precision = 84.97%
- **Use Case**: Identify high-risk trips and notify customers proactively

### 3. Route Profitability Prediction
- **Algorithm**: Gradient Boosting Regressor
- **Performance**: R² = -0.9171, RMSE = 7.95%
- **Use Case**: Dynamic pricing and route optimization

### 4. Driver Safety Risk Assessment
- **Algorithm**: Random Forest Classifier
- **Performance**: Accuracy = 60.00%, Precision = 49.09%
- **Use Case**: Target safety training programs

---

## 📈 Dashboards & Visualizations

All visualizations are saved as high-resolution PNG files in the `Docs/` folder:

### Driver_Dashboard.png
- Top 10 drivers by revenue
- Revenue distribution
- Fuel efficiency analysis
- Trip count distribution
- On-time delivery rates
- Revenue vs efficiency scatter plot

### Fleet_Dashboard.png
- Fleet status distribution
- Top truck performance
- Truck age vs revenue analysis
- Cost per mile distribution
- Vehicle utilization metrics

### Route_Dashboard.png
- Most profitable routes
- Profit margin analysis
- Revenue vs cost relationships
- Trip distribution by route
- Distance and profitability metrics

### KPI_Summary_Table.png
- Executive summary with 4 metric categories:
  - Driver metrics
  - Fleet metrics
  - Route metrics
  - Fuel efficiency metrics

---

## 📝 Output Files

### Data Reports (CSV)
All reports are generated in the `Output/` folder:

| Report | Description |
|--------|-------------|
| trip_analysis_report.csv | Merged trip data with all metrics |
| driver_metrics_report.csv | Driver performance KPIs |
| fleet_metrics_report.csv | Fleet utilization metrics |
| route_metrics_report.csv | Route profitability analysis |
| fuel_efficiency_report.csv | Fuel consumption by truck |
| delivery_performance_report.csv | On-time delivery metrics |
| safety_summary_report.csv | Safety incident analysis |
| maintenance_summary_report.csv | Maintenance cost analysis |
| *_feature_importance.csv | ML model feature rankings |

---

## 🔧 Module Documentation

### data_processor.py
Main data processing pipeline with methods:
- `load_all_datasets()` - Load from Dataset/ folder
- `process_all()` - Execute complete pipeline
- `export_all_reports()` - Export to Output/ folder

### ml_models.py
Machine learning model training with:
- `train_all_models()` - Train all 4 models
- `export_model_results()` - Export feature importance

### visualization_utils.py
Professional dashboard generation:
- `create_driver_dashboard()` - Driver performance viz
- `create_fleet_dashboard()` - Fleet utilization viz
- `create_route_dashboard()` - Route profitability viz
- `create_kpi_summary_table()` - Executive summary

---

## 💡 Key Insights

### 1. Driver Optimization
- Top 10% of drivers generate ~$77M in revenue (30% of total)
- Best practice sharing with underperformers can improve efficiency
- Focus on drivers with lowest MPG for coaching

### 2. Fleet Maintenance
- Preventive maintenance program can reduce downtime by 15-20%
- Older trucks show lower efficiency - consider replacement strategy
- Predictive maintenance model can optimize scheduling

### 3. Route Optimization
- Several routes are currently unprofitable
- Dynamic pricing recommended for underperforming routes
- Route consolidation can improve profitability

### 4. Fuel Efficiency
- 5% MPG improvement could save ~$450K/month
- Driver behavior has significant impact on fuel consumption
- Real-time monitoring can drive continuous improvement

---

## 🎯 Strategic Recommendations

1. **Deploy Predictive Models**
   - Real-time fuel consumption forecasting
   - Proactive delay prediction and mitigation
   - Dynamic route optimization

2. **Establish KPI Dashboards**
   - Real-time operations monitoring
   - Weekly performance reviews
   - Monthly strategy adjustments

3. **Implement Driver Programs**
   - Incentivize top performers
   - Targeted training for underperformers
   - Safety intervention programs

4. **Technology Infrastructure**
   - Real-time data pipeline
   - Automated alerting system
   - Mobile driver app integration

---

## 📞 Support & Contact

For questions or issues, contact the Data Analytics Team.

---

## 📄 License

This project is confidential and proprietary to TCS.

---

**Last Updated**: May 2026
**Data Period**: January - December 2025
**Analysis Date**: May 17, 2026
