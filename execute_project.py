"""
TCS Internship Project - Main Execution Script
==============================================
Complete end-to-end data pipeline execution

Author: Data Analytics Team
Date: April 2024
"""

import sys
import os
from datetime import datetime

# Set UTF-8 encoding for Windows systems
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project path
project_path = 'e:/All-Prog/Data-Visualization/TCS Interhsip - (Project)/'
sys.path.append(project_path)

from data_processor import LogisticsDataProcessor
from ml_models import LogisticsMLModels
from visualization_utils import LogisticsVisualizer
import pandas as pd
import matplotlib.pyplot as plt


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def main():
    """Execute complete TCS project pipeline"""
    
    print_header("TCS INTERNSHIP PROJECT - COMPLETE EXECUTION")
    
    # 1. DATA PROCESSING
    print_header("STEP 1: DATA PROCESSING AND PREPARATION")
    
    data_path = project_path
    processor = LogisticsDataProcessor(data_path)
    processed_data = processor.process_all()
    
    # 2. MACHINE LEARNING MODELS
    print_header("STEP 2: MACHINE LEARNING MODEL TRAINING")
    
    ml_models = LogisticsMLModels(processor)
    trained_models = ml_models.train_all_models()
    
    # 3. CREATE VISUALIZATIONS
    print_header("STEP 3: CREATING PROFESSIONAL DASHBOARDS")
    
    print("Generating visualizations...")
    
    driver_metrics = processed_data['driver_metrics']
    fleet_metrics = processed_data['fleet_metrics']
    route_metrics = processed_data['route_metrics']
    fuel_efficiency = processed_data['fuel_efficiency']
    
    print("  - Creating driver performance dashboard...")
    LogisticsVisualizer.create_driver_dashboard(driver_metrics)
    plt.close()
    
    print("  - Creating fleet utilization dashboard...")
    LogisticsVisualizer.create_fleet_dashboard(fleet_metrics)
    plt.close()
    
    print("  - Creating route profitability dashboard...")
    LogisticsVisualizer.create_route_dashboard(route_metrics)
    plt.close()
    
    print("  - Creating KPI summary table...")
    LogisticsVisualizer.create_kpi_summary_table(driver_metrics, fleet_metrics, route_metrics, fuel_efficiency)
    plt.close()
    
    # 4. EXPORT REPORTS
    print_header("STEP 4: EXPORTING REPORTS")
    
    print("Exporting processed data reports...")
    processor.export_all_reports(data_path)
    
    print("\nExporting model results...")
    ml_models.export_model_results(data_path)
    
    # 5. GENERATE INSIGHTS
    print_header("STEP 5: KEY INSIGHTS AND ANALYSIS")
    
    generate_insights(processed_data, ml_models)
    
    # 6. FINAL SUMMARY
    print_header("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    
    print("GENERATED OUTPUTS:")
    print("-" * 60)
    print("\nData Reports (CSV):")
    print("  ✓ driver_metrics_report.csv")
    print("  ✓ fleet_metrics_report.csv")
    print("  ✓ route_metrics_report.csv")
    print("  ✓ fuel_efficiency_report.csv")
    print("  ✓ delivery_performance_report.csv")
    print("  ✓ safety_summary_report.csv")
    print("  ✓ maintenance_summary_report.csv")
    
    print("\nVisualizations (PNG):")
    print("  ✓ Driver_Dashboard.png")
    print("  ✓ Fleet_Dashboard.png")
    print("  ✓ Route_Dashboard.png")
    print("  ✓ KPI_Summary_Table.png")
    
    print("\nMachine Learning Models:")
    print("  ✓ Fuel Consumption Prediction")
    print("  ✓ Delivery Delay Forecasting")
    print("  ✓ Route Profitability Prediction")
    print("  ✓ Driver Safety Risk Assessment")
    
    print("\nModel Results (CSV):")
    print("  ✓ fuel_consumption_feature_importance.csv")
    print("  ✓ delivery_delay_feature_importance.csv")
    print("  ✓ route_profitability_feature_importance.csv")
    print("  ✓ driver_safety_feature_importance.csv")
    
    print("\n" + "="*80)
    print("Execution Time: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("="*80)


def generate_insights(processed_data, ml_models):
    """Generate and display key insights"""
    
    driver_metrics = processed_data['driver_metrics']
    fleet_metrics = processed_data['fleet_metrics']
    route_metrics = processed_data['route_metrics']
    fuel_efficiency = processed_data['fuel_efficiency']
    
    print("\n1. DRIVER PERFORMANCE INSIGHTS:")
    print("-" * 60)
    best_driver = driver_metrics.loc[driver_metrics['total_revenue'].idxmax()]
    print(f"  Top Performer: {best_driver['first_name']} {best_driver['last_name']}")
    print(f"     Revenue: ${best_driver['total_revenue']:,.2f}")
    print(f"     Trips: {best_driver['trips_count']:.0f}")
    print(f"     MPG: {best_driver['avg_mpg']:.2f}")
    print(f"     On-Time Rate: {best_driver['on_time_delivery_rate']*100:.1f}%")
    
    print(f"\n  Fleet Averages:")
    print(f"     Total Revenue: ${driver_metrics['total_revenue'].sum():,.2f}")
    print(f"     Average Driver Revenue: ${driver_metrics['total_revenue'].mean():,.2f}")
    print(f"     Average Trips: {driver_metrics['trips_count'].mean():.0f}")
    print(f"     Average MPG: {driver_metrics['avg_mpg'].mean():.2f}")
    
    print("\n2. FLEET UTILIZATION INSIGHTS:")
    print("-" * 60)
    active_trucks = len(fleet_metrics[fleet_metrics['status'] == 'Active'])
    print(f"  Fleet Status: {active_trucks}/{len(fleet_metrics)} trucks active")
    print(f"     Total Revenue: ${fleet_metrics['total_revenue'].sum():,.2f}")
    print(f"     Average Truck Revenue: ${fleet_metrics['total_revenue'].mean():,.2f}")
    print(f"     Average Cost per Mile: ${fleet_metrics['cost_per_mile'].mean():.2f}")
    
    best_truck = fleet_metrics.loc[fleet_metrics['total_revenue'].idxmax()]
    print(f"\n  Top Truck: {best_truck['truck_id']} ({best_truck['make']})")
    print(f"     Total Revenue: ${best_truck['total_revenue']:,.2f}")
    print(f"     Total Miles: {best_truck['total_miles']:,.0f}")
    
    print("\n3. ROUTE PROFITABILITY INSIGHTS:")
    print("-" * 60)
    print(f"  Total Routes: {len(route_metrics)}")
    print(f"     Total Profit: ${route_metrics['profit'].sum():,.2f}")
    print(f"     Average Profit per Route: ${route_metrics['profit'].mean():,.2f}")
    print(f"     Average Profit Margin: {route_metrics['profit_margin'].mean():.2f}%")
    
    best_route = route_metrics.loc[route_metrics['profit'].idxmax()]
    worst_route = route_metrics.loc[route_metrics['profit'].idxmin()]
    print(f"\n  Most Profitable Route: {best_route['route_id']}")
    print(f"     Profit: ${best_route['profit']:,.2f}")
    print(f"     Margin: {best_route['profit_margin']:.2f}%")
    print(f"\n  Least Profitable Route: {worst_route['route_id']}")
    print(f"     Profit: ${worst_route['profit']:,.2f}")
    print(f"     Margin: {worst_route['profit_margin']:.2f}%")
    
    print("\n4. FUEL EFFICIENCY INSIGHTS:")
    print("-" * 60)
    print(f"  Fleet Average MPG: {fuel_efficiency['calculated_mpg'].mean():.2f}")
    print(f"     Total Fuel Cost: ${fuel_efficiency['fuel_cost'].sum():,.2f}")
    print(f"     Average Cost per Mile: ${fuel_efficiency['cost_per_mile'].mean():.2f}")
    
    best_mpg = fuel_efficiency.loc[fuel_efficiency['calculated_mpg'].idxmax()]
    worst_mpg = fuel_efficiency.loc[fuel_efficiency['calculated_mpg'].idxmin()]
    print(f"\n  Most Efficient: {best_mpg['truck_id']} - {best_mpg['calculated_mpg']:.2f} MPG")
    print(f"  Least Efficient: {worst_mpg['truck_id']} - {worst_mpg['calculated_mpg']:.2f} MPG")
    
    print("\n5. MACHINE LEARNING MODEL PERFORMANCE:")
    print("-" * 60)
    for model_name, results in ml_models.results.items():
        print(f"\n  {model_name.upper()}:")
        if 'r2' in results:
            print(f"     R-Squared Score: {results['r2']:.4f}")
        if 'rmse' in results:
            print(f"     RMSE: {results['rmse']:.4f}")
        if 'accuracy' in results:
            print(f"     Accuracy: {results['accuracy']:.2%}")
    
    print("\n6. STRATEGIC RECOMMENDATIONS:")
    print("-" * 60)
    print("  ✓ Implement predictive fuel consumption model for cost optimization")
    print("  ✓ Use delivery delay predictions for proactive customer communication")
    print("  ✓ Apply route profitability model for dynamic pricing")
    print("  ✓ Deploy driver safety model for targeted training programs")
    print("  ✓ Establish real-time KPI dashboards for operations team")
    print("  ✓ Review and optimize underperforming routes")
    print("  ✓ Implement preventive maintenance for high-idle trucks")
    print("  ✓ Share best practices from top performers with fleet")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
