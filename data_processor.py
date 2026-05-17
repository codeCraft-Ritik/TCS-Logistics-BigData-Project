"""
TCS Internship Project - Data Processing Pipeline
================================================
Data cleaning, transformation, and aggregation for Logistics Analytics

Author: Data Analytics Team
Date: April 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

# Set UTF-8 encoding for Windows systems
os.environ['PYTHONIOENCODING'] = 'utf-8'

warnings.filterwarnings('ignore')


class LogisticsDataProcessor:
    """Main data processing class for logistics data pipeline"""
    
    def __init__(self, data_path):
        """Initialize processor with data path"""
        self.data_path = data_path
        self.data = {}
        self.processed_data = {}
        
    def load_all_datasets(self):
        """Load all CSV files from data path"""
        print("Loading all datasets...")
        try:
            dataset_path = self.data_path + 'Dataset/'
            self.data['drivers'] = pd.read_csv(dataset_path + 'drivers.csv')
            self.data['trucks'] = pd.read_csv(dataset_path + 'trucks.csv')
            self.data['trailers'] = pd.read_csv(dataset_path + 'trailers.csv')
            self.data['routes'] = pd.read_csv(dataset_path + 'routes.csv')
            self.data['loads'] = pd.read_csv(dataset_path + 'loads.csv')
            self.data['trips'] = pd.read_csv(dataset_path + 'trips.csv')
            self.data['customers'] = pd.read_csv(dataset_path + 'customers.csv')
            self.data['facilities'] = pd.read_csv(dataset_path + 'facilities.csv')
            self.data['fuel_purchases'] = pd.read_csv(dataset_path + 'fuel_purchases.csv')
            self.data['maintenance'] = pd.read_csv(dataset_path + 'maintenance_records.csv')
            self.data['delivery_events'] = pd.read_csv(dataset_path + 'delivery_events.csv')
            self.data['safety_incidents'] = pd.read_csv(dataset_path + 'safety_incidents.csv')
            self.data['driver_metrics'] = pd.read_csv(dataset_path + 'driver_monthly_metrics.csv')
            self.data['truck_metrics'] = pd.read_csv(dataset_path + 'truck_utilization_metrics.csv')
            
            print(f"✓ All datasets loaded successfully!")
            return self.data
        except Exception as e:
            print(f"✗ Error loading datasets: {e}")
            raise
    
    def convert_date_columns(self):
        """Convert date columns to datetime format"""
        print("Converting date columns...")
        
        # Drivers
        date_cols = ['hire_date', 'date_of_birth', 'termination_date']
        for col in date_cols:
            if col in self.data['drivers'].columns:
                self.data['drivers'][col] = pd.to_datetime(self.data['drivers'][col], errors='coerce')
        
        # Trucks
        if 'acquisition_date' in self.data['trucks'].columns:
            self.data['trucks']['acquisition_date'] = pd.to_datetime(
                self.data['trucks']['acquisition_date'], errors='coerce'
            )
        
        # Trips
        if 'dispatch_date' in self.data['trips'].columns:
            self.data['trips']['dispatch_date'] = pd.to_datetime(
                self.data['trips']['dispatch_date'], errors='coerce'
            )
        
        # Loads
        load_date_cols = ['booking_date', 'scheduled_pickup', 'scheduled_delivery']
        for col in load_date_cols:
            if col in self.data['loads'].columns:
                self.data['loads'][col] = pd.to_datetime(self.data['loads'][col], errors='coerce')
        
        # Fuel Purchases
        if 'purchase_date' in self.data['fuel_purchases'].columns:
            self.data['fuel_purchases']['purchase_date'] = pd.to_datetime(
                self.data['fuel_purchases']['purchase_date'], errors='coerce'
            )
        
        # Maintenance
        maint_date_cols = ['maintenance_date', 'completion_date']
        for col in maint_date_cols:
            if col in self.data['maintenance'].columns:
                self.data['maintenance'][col] = pd.to_datetime(
                    self.data['maintenance'][col], errors='coerce'
                )
        
        # Delivery Events
        delivery_date_cols = ['actual_pickup_time', 'actual_delivery_time', 
                             'scheduled_pickup_time', 'scheduled_delivery_time']
        for col in delivery_date_cols:
            if col in self.data['delivery_events'].columns:
                self.data['delivery_events'][col] = pd.to_datetime(
                    self.data['delivery_events'][col], errors='coerce'
                )
        
        print("✓ Date columns converted successfully!")
    
    def clean_data(self):
        """Clean and validate data"""
        print("Cleaning data...")
        
        # Remove duplicates
        for key in self.data:
            initial_len = len(self.data[key])
            self.data[key] = self.data[key].drop_duplicates()
            if len(self.data[key]) < initial_len:
                print(f"  - Removed {initial_len - len(self.data[key])} duplicates from {key}")
        
        # Handle missing values
        for key in self.data:
            missing_count = self.data[key].isnull().sum().sum()
            if missing_count > 0:
                print(f"  - {key}: {missing_count} missing values")
        
        print("✓ Data cleaning completed!")
    
    def merge_trip_analysis_dataset(self):
        """Create comprehensive trip analysis dataset"""
        print("Creating comprehensive trip analysis dataset...")
        
        trip_data = self.data['trips'].copy()
        
        # Merge with driver info
        trip_data = trip_data.merge(
            self.data['drivers'][['driver_id', 'first_name', 'last_name', 'cdl_class', 'years_experience']],
            on='driver_id', how='left'
        )
        
        # Merge with truck info
        trip_data = trip_data.merge(
            self.data['trucks'][['truck_id', 'make', 'model_year', 'fuel_type', 'status']],
            on='truck_id', how='left'
        )
        
        # Merge with load info
        trip_data = trip_data.merge(
            self.data['loads'][['load_id', 'customer_id', 'route_id', 'revenue', 'booking_type']],
            on='load_id', how='left'
        )
        
        # Merge with route info
        trip_data = trip_data.merge(
            self.data['routes'][['route_id', 'typical_distance_miles', 'base_rate_per_mile', 'fuel_surcharge_rate']],
            on='route_id', how='left'
        )
        
        # Merge with customer info
        trip_data = trip_data.merge(
            self.data['customers'][['customer_id', 'customer_name', 'customer_type']],
            on='customer_id', how='left'
        )
        
        self.processed_data['trip_analysis'] = trip_data
        print(f"✓ Trip analysis dataset created with {len(trip_data)} records")
        return trip_data
    
    def calculate_driver_metrics(self, fuel_price=3.5):
        """Calculate driver performance metrics"""
        print("Calculating driver performance metrics...")
        
        trip_data = self.processed_data.get('trip_analysis')
        if trip_data is None:
            trip_data = self.merge_trip_analysis_dataset()
        
        driver_perf = trip_data.groupby(['driver_id', 'first_name', 'last_name']).agg({
            'trip_id': 'count',
            'actual_distance_miles': 'sum',
            'fuel_gallons_used': 'sum',
            'average_mpg': 'mean',
            'actual_duration_hours': 'sum',
            'idle_time_hours': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        driver_perf.columns = ['driver_id', 'first_name', 'last_name', 'trips_count',
                               'total_miles', 'total_fuel', 'avg_mpg', 'total_hours',
                               'total_idle_hours', 'total_revenue']
        
        # Calculate derived metrics
        driver_perf['revenue_per_mile'] = driver_perf['total_revenue'] / driver_perf['total_miles']
        driver_perf['fuel_cost_per_mile'] = (driver_perf['total_fuel'] * fuel_price) / driver_perf['total_miles']
        driver_perf['miles_per_hour'] = driver_perf['total_miles'] / driver_perf['total_hours']
        driver_perf['idle_percentage'] = (driver_perf['total_idle_hours'] / driver_perf['total_hours']) * 100
        
        # Add on-time delivery rate
        if len(self.data['delivery_events']) > 0:
            on_time = self.data['delivery_events'].groupby('trip_id')['on_time_flag'].mean().reset_index()
            on_time_driver = on_time.merge(self.data['trips'][['trip_id', 'driver_id']], on='trip_id')
            on_time_driver = on_time_driver.groupby('driver_id')['on_time_flag'].mean().reset_index()
            on_time_driver.columns = ['driver_id', 'on_time_delivery_rate']
            driver_perf = driver_perf.merge(on_time_driver, on='driver_id', how='left')
        
        self.processed_data['driver_metrics'] = driver_perf
        print(f"✓ Driver metrics calculated for {len(driver_perf)} drivers")
        return driver_perf.sort_values('total_revenue', ascending=False)
    
    def calculate_fleet_metrics(self, fuel_price=3.5):
        """Calculate fleet utilization metrics"""
        print("Calculating fleet utilization metrics...")
        
        trip_data = self.processed_data.get('trip_analysis')
        if trip_data is None:
            trip_data = self.merge_trip_analysis_dataset()
        
        fleet_util = trip_data.groupby(['truck_id', 'make', 'model_year', 'status']).agg({
            'trip_id': 'count',
            'actual_distance_miles': 'sum',
            'fuel_gallons_used': 'sum',
            'average_mpg': 'mean',
            'actual_duration_hours': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        fleet_util.columns = ['truck_id', 'make', 'model_year', 'status', 'trips_count',
                              'total_miles', 'total_fuel', 'avg_mpg', 'total_hours', 'total_revenue']
        
        # Calculate derived metrics
        fleet_util['cost_per_mile'] = (fleet_util['total_fuel'] * fuel_price) / fleet_util['total_miles']
        fleet_util['miles_per_trip'] = fleet_util['total_miles'] / fleet_util['trips_count']
        fleet_util['revenue_per_mile'] = fleet_util['total_revenue'] / fleet_util['total_miles']
        
        self.processed_data['fleet_metrics'] = fleet_util
        print(f"✓ Fleet metrics calculated for {len(fleet_util)} trucks")
        return fleet_util.sort_values('total_revenue', ascending=False)
    
    def calculate_route_metrics(self, fuel_price=3.5, overhead_multiplier=1.3):
        """Calculate route profitability metrics"""
        print("Calculating route profitability metrics...")
        
        trip_data = self.processed_data.get('trip_analysis')
        if trip_data is None:
            trip_data = self.merge_trip_analysis_dataset()
        
        route_profit = trip_data.groupby('route_id').agg({
            'trip_id': 'count',
            'actual_distance_miles': 'sum',
            'fuel_gallons_used': 'sum',
            'average_mpg': 'mean',
            'revenue': 'sum'
        }).reset_index()
        
        route_profit.columns = ['route_id', 'trips_count',
                                'total_distance', 'total_fuel', 'avg_mpg', 'total_revenue']
        
        # Calculate profitability
        route_profit['fuel_cost'] = route_profit['total_fuel'] * fuel_price
        route_profit['estimated_operating_cost'] = route_profit['fuel_cost'] * overhead_multiplier
        route_profit['profit'] = route_profit['total_revenue'] - route_profit['estimated_operating_cost']
        route_profit['profit_margin'] = (route_profit['profit'] / route_profit['total_revenue']) * 100
        route_profit['cost_per_mile'] = route_profit['estimated_operating_cost'] / route_profit['total_distance']
        route_profit['revenue_per_mile'] = route_profit['total_revenue'] / route_profit['total_distance']
        
        self.processed_data['route_metrics'] = route_profit
        print(f"✓ Route metrics calculated for {len(route_profit)} routes")
        return route_profit.sort_values('profit', ascending=False)
    
    def calculate_fuel_efficiency(self):
        """Calculate fuel efficiency metrics"""
        print("Calculating fuel efficiency metrics...")
        
        trip_data = self.processed_data.get('trip_analysis')
        if trip_data is None:
            trip_data = self.merge_trip_analysis_dataset()
        
        fuel_eff = trip_data.groupby(['truck_id', 'make']).agg({
            'trip_id': 'count',
            'actual_distance_miles': 'sum',
            'fuel_gallons_used': 'sum',
            'average_mpg': 'mean'
        }).reset_index()
        
        fuel_eff.columns = ['truck_id', 'make', 'trips', 'total_miles', 'total_fuel', 'avg_mpg']
        
        fuel_eff['calculated_mpg'] = fuel_eff['total_miles'] / fuel_eff['total_fuel']
        fuel_eff['fuel_cost'] = fuel_eff['total_fuel'] * 3.5
        fuel_eff['cost_per_mile'] = fuel_eff['fuel_cost'] / fuel_eff['total_miles']
        
        self.processed_data['fuel_efficiency'] = fuel_eff
        print(f"✓ Fuel efficiency calculated for {len(fuel_eff)} trucks")
        return fuel_eff.sort_values('calculated_mpg', ascending=False)
    
    def get_delivery_performance(self):
        """Get delivery performance metrics"""
        print("Calculating delivery performance...")
        
        if len(self.data['delivery_events']) == 0:
            print("⚠ No delivery events data available")
            return None
        
        delivery_perf = self.data['delivery_events'].groupby('trip_id').agg({
            'on_time_flag': 'mean',
            'detention_minutes': 'sum'
        }).reset_index()
        
        delivery_perf = delivery_perf.merge(self.data['trips'][['trip_id', 'driver_id', 'truck_id']], on='trip_id')
        
        on_time_rate = (delivery_perf['on_time_flag'].sum() / len(delivery_perf)) * 100
        avg_detention = delivery_perf['detention_minutes'].mean()
        
        print(f"✓ Delivery performance calculated")
        print(f"  - On-time delivery rate: {on_time_rate:.2f}%")
        print(f"  - Average detention: {avg_detention:.2f} minutes")
        
        self.processed_data['delivery_performance'] = delivery_perf
        return delivery_perf
    
    def get_safety_summary(self):
        """Get safety incidents summary"""
        if len(self.data['safety_incidents']) == 0:
            print("⚠ No safety incidents data available")
            return None
        
        print("Summarizing safety incidents...")
        
        incidents = self.data['safety_incidents']
        total_incidents = len(incidents)
        preventable = incidents['preventable_flag'].sum()
        total_damage = (incidents['vehicle_damage_cost'] + incidents['cargo_damage_cost']).sum()
        
        driver_incidents = incidents.groupby('driver_id').agg({
            'incident_id': 'count',
            'vehicle_damage_cost': 'sum',
            'cargo_damage_cost': 'sum',
            'preventable_flag': 'sum'
        }).reset_index()
        driver_incidents.columns = ['driver_id', 'total_incidents', 'vehicle_damage', 'cargo_damage', 'preventable_incidents']
        driver_incidents['total_damage'] = driver_incidents['vehicle_damage'] + driver_incidents['cargo_damage']
        
        print(f"✓ Safety summary completed")
        print(f"  - Total incidents: {total_incidents}")
        print(f"  - Preventable incidents: {int(preventable)}")
        print(f"  - Total damage cost: ${total_damage:,.2f}")
        
        self.processed_data['safety_summary'] = driver_incidents
        return driver_incidents
    
    def get_maintenance_summary(self):
        """Get maintenance summary"""
        if len(self.data['maintenance']) == 0:
            print("⚠ No maintenance data available")
            return None
        
        print("Summarizing maintenance records...")
        
        maint = self.data['maintenance'].copy()
        maint['maintenance_date'] = pd.to_datetime(maint['maintenance_date'], errors='coerce')
        
        maint_summary = maint.groupby('truck_id').agg({
            'maintenance_id': 'count',
            'labor_hours': 'sum',
            'parts_cost': 'sum',
            'total_cost': 'sum',
            'downtime_hours': 'sum'
        }).reset_index()
        maint_summary.columns = ['truck_id', 'maintenance_count', 'total_labor_hours', 'total_parts_cost', 'total_maintenance_cost', 'total_downtime_hours']
        maint_summary['cost_per_maintenance'] = maint_summary['total_maintenance_cost'] / maint_summary['maintenance_count']
        
        print(f"✓ Maintenance summary completed")
        print(f"  - Total records: {len(maint)}")
        print(f"  - Total labor hours: {maint['labor_hours'].sum():,.0f}")
        print(f"  - Total maintenance cost: ${maint['total_cost'].sum():,.2f}")
        
        self.processed_data['maintenance_summary'] = maint_summary
        return maint_summary.sort_values('maintenance_count', ascending=False)
    
    def process_all(self):
        """Run complete data processing pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPLETE DATA PROCESSING PIPELINE".center(80))
        print("="*80 + "\n")
        
        self.load_all_datasets()
        self.convert_date_columns()
        self.clean_data()
        self.merge_trip_analysis_dataset()
        self.calculate_driver_metrics()
        self.calculate_fleet_metrics()
        self.calculate_route_metrics()
        self.calculate_fuel_efficiency()
        self.get_delivery_performance()
        self.get_safety_summary()
        self.get_maintenance_summary()
        
        print("\n" + "="*80)
        print("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY".center(80))
        print("="*80 + "\n")
        
        return self.processed_data
    
    def export_all_reports(self, output_path='./'):
        """Export all processed data to CSV files"""
        final_output_path = output_path + 'Output/'
        print(f"\nExporting reports to {final_output_path}...")
        
        for name, data in self.processed_data.items():
            if data is not None:
                filename = f"{final_output_path}{name}_report.csv"
                data.to_csv(filename, index=False)
                print(f"✓ Exported {filename}")


if __name__ == "__main__":
    # Example usage
    data_path = 'e:/All-Prog/Data-Visualization/TCS Interhsip - (Project)/'
    processor = LogisticsDataProcessor(data_path)
    
    # Run complete pipeline
    processed = processor.process_all()
    
    # Export reports to Output folder
    processor.export_all_reports(data_path)
