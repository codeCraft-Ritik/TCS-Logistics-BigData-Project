"""
Visualization Utilities for TCS Logistics Project
==================================================
Professional dashboard and visualization generation

Author: Data Analytics Team
Date: May 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

# Set UTF-8 encoding for Windows systems
os.environ['PYTHONIOENCODING'] = 'utf-8'


class LogisticsVisualizer:
    """Professional visualization utilities for logistics analytics"""
    
    @staticmethod
    def create_driver_dashboard(driver_metrics):
        """Create comprehensive driver performance dashboard"""
        
        print("Creating driver performance dashboard...")
        
        # Create figure with GridSpec for complex layout
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Top Drivers by Revenue
        ax1 = fig.add_subplot(gs[0, :2])
        top_drivers = driver_metrics.nlargest(10, 'total_revenue')
        ax1.barh(range(len(top_drivers)), top_drivers['total_revenue'], color='steelblue', edgecolor='black')
        ax1.set_yticks(range(len(top_drivers)))
        ax1.set_yticklabels([f"{row['first_name']} {row['last_name']}" for _, row in top_drivers.iterrows()], fontsize=9)
        ax1.set_title('Top 10 Drivers by Revenue', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Total Revenue ($)', fontsize=10)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Revenue Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(driver_metrics['total_revenue'], bins=20, color='teal', edgecolor='black', alpha=0.7)
        ax2.set_title('Revenue Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Revenue ($)', fontsize=10)
        ax2.set_ylabel('Number of Drivers', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. MPG Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(driver_metrics['avg_mpg'], bins=20, color='orange', edgecolor='black', alpha=0.7)
        ax3.set_title('Fuel Efficiency Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Average MPG', fontsize=10)
        ax3.set_ylabel('Number of Drivers', fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Trips Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(driver_metrics['trips_count'], bins=20, color='green', edgecolor='black', alpha=0.7)
        ax4.set_title('Trip Count Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Number of Trips', fontsize=10)
        ax4.set_ylabel('Number of Drivers', fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. On-Time Delivery Rate
        ax5 = fig.add_subplot(gs[1, 2])
        on_time_data = driver_metrics['on_time_delivery_rate'].fillna(0)
        ax5.hist(on_time_data * 100, bins=20, color='purple', edgecolor='black', alpha=0.7)
        ax5.set_title('On-Time Delivery Rate', fontsize=12, fontweight='bold')
        ax5.set_xlabel('On-Time Rate (%)', fontsize=10)
        ax5.set_ylabel('Number of Drivers', fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Revenue vs MPG Scatter
        ax6 = fig.add_subplot(gs[2, 0])
        scatter = ax6.scatter(driver_metrics['avg_mpg'], driver_metrics['total_revenue'], 
                            c=driver_metrics['trips_count'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        ax6.set_title('Revenue vs Fuel Efficiency', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Average MPG', fontsize=10)
        ax6.set_ylabel('Total Revenue ($)', fontsize=10)
        plt.colorbar(scatter, ax=ax6, label='Trip Count')
        ax6.grid(alpha=0.3)
        
        # 7. Idle Time Analysis
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.hist(driver_metrics['idle_percentage'], bins=20, color='red', edgecolor='black', alpha=0.7)
        ax7.set_title('Idle Time Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Idle Time (%)', fontsize=10)
        ax7.set_ylabel('Number of Drivers', fontsize=10)
        ax7.grid(axis='y', alpha=0.3)
        
        # 8. Revenue per Mile
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.hist(driver_metrics['revenue_per_mile'], bins=20, color='brown', edgecolor='black', alpha=0.7)
        ax8.set_title('Revenue per Mile Distribution', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Revenue per Mile ($)', fontsize=10)
        ax8.set_ylabel('Number of Drivers', fontsize=10)
        ax8.grid(axis='y', alpha=0.3)
        
        plt.suptitle('DRIVER PERFORMANCE DASHBOARD', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('Docs/Driver_Dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Driver dashboard saved to Docs/Driver_Dashboard.png")
        return fig
    
    @staticmethod
    def create_fleet_dashboard(fleet_metrics):
        """Create comprehensive fleet utilization dashboard"""
        
        print("Creating fleet utilization dashboard...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Fleet Status Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        status_counts = fleet_metrics['status'].value_counts()
        colors = ['green', 'orange', 'red', 'gray'][:len(status_counts)]
        ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Fleet Status Distribution', fontsize=12, fontweight='bold')
        
        # 2. Top Trucks by Revenue
        ax2 = fig.add_subplot(gs[0, 1:])
        top_trucks = fleet_metrics.nlargest(10, 'total_revenue')
        ax2.barh(range(len(top_trucks)), top_trucks['total_revenue'], color='steelblue', edgecolor='black')
        ax2.set_yticks(range(len(top_trucks)))
        ax2.set_yticklabels(top_trucks['truck_id'], fontsize=9)
        ax2.set_title('Top 10 Trucks by Revenue', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Total Revenue ($)', fontsize=10)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Truck Age vs Revenue
        ax3 = fig.add_subplot(gs[1, 0])
        fleet_metrics['truck_age'] = 2026 - fleet_metrics['model_year']
        scatter = ax3.scatter(fleet_metrics['truck_age'], fleet_metrics['total_revenue'], 
                            c=fleet_metrics['trips_count'], cmap='plasma', s=100, alpha=0.6, edgecolors='black')
        ax3.set_title('Truck Age vs Revenue', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Truck Age (years)', fontsize=10)
        ax3.set_ylabel('Total Revenue ($)', fontsize=10)
        plt.colorbar(scatter, ax=ax3, label='Trip Count')
        ax3.grid(alpha=0.3)
        
        # 4. Cost per Mile Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(fleet_metrics['cost_per_mile'], bins=20, color='red', edgecolor='black', alpha=0.7)
        ax4.set_title('Cost per Mile Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Cost per Mile ($)', fontsize=10)
        ax4.set_ylabel('Number of Trucks', fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Miles per Trip
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(fleet_metrics['miles_per_trip'], bins=20, color='purple', edgecolor='black', alpha=0.7)
        ax5.set_title('Miles per Trip Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Miles per Trip', fontsize=10)
        ax5.set_ylabel('Number of Trucks', fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Revenue Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(fleet_metrics['total_revenue'], bins=20, color='teal', edgecolor='black', alpha=0.7)
        ax6.set_title('Revenue Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Total Revenue ($)', fontsize=10)
        ax6.set_ylabel('Number of Trucks', fontsize=10)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Average MPG Distribution
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.hist(fleet_metrics['avg_mpg'], bins=20, color='orange', edgecolor='black', alpha=0.7)
        ax7.set_title('Average MPG Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Average MPG', fontsize=10)
        ax7.set_ylabel('Number of Trucks', fontsize=10)
        ax7.grid(axis='y', alpha=0.3)
        
        # 8. Trips Distribution
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.hist(fleet_metrics['trips_count'], bins=20, color='green', edgecolor='black', alpha=0.7)
        ax8.set_title('Trip Count Distribution', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Number of Trips', fontsize=10)
        ax8.set_ylabel('Number of Trucks', fontsize=10)
        ax8.grid(axis='y', alpha=0.3)
        
        plt.suptitle('FLEET UTILIZATION DASHBOARD', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('Docs/Fleet_Dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Fleet dashboard saved to Docs/Fleet_Dashboard.png")
        return fig
    
    @staticmethod
    def create_route_dashboard(route_metrics):
        """Create comprehensive route profitability dashboard"""
        
        print("Creating route profitability dashboard...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Top Profitable Routes
        ax1 = fig.add_subplot(gs[0, :2])
        top_routes = route_metrics.nlargest(10, 'profit')
        ax1.barh(range(len(top_routes)), top_routes['profit'], color='darkgreen', edgecolor='black')
        ax1.set_yticks(range(len(top_routes)))
        ax1.set_yticklabels(top_routes['route_id'], fontsize=9)
        ax1.set_title('Top 10 Most Profitable Routes', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Profit ($)', fontsize=10)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Profit Margin Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(route_metrics['profit_margin'], bins=20, color='green', edgecolor='black', alpha=0.7)
        ax2.set_title('Profit Margin Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Profit Margin (%)', fontsize=10)
        ax2.set_ylabel('Number of Routes', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Revenue vs Cost
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(route_metrics['estimated_operating_cost'], route_metrics['total_revenue'],
                            c=route_metrics['profit'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
        ax3.set_title('Revenue vs Operating Cost', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Operating Cost ($)', fontsize=10)
        ax3.set_ylabel('Revenue ($)', fontsize=10)
        plt.colorbar(scatter, ax=ax3, label='Profit ($)')
        ax3.grid(alpha=0.3)
        
        # 4. Revenue per Mile vs Cost per Mile
        ax4 = fig.add_subplot(gs[1, 1])
        scatter = ax4.scatter(route_metrics['cost_per_mile'], route_metrics['revenue_per_mile'],
                            c=route_metrics['trips_count'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        ax4.set_title('Revenue vs Cost per Mile', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Cost per Mile ($)', fontsize=10)
        ax4.set_ylabel('Revenue per Mile ($)', fontsize=10)
        plt.colorbar(scatter, ax=ax4, label='Trip Count')
        ax4.grid(alpha=0.3)
        
        # 5. Trip Count Distribution
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(route_metrics['trips_count'], bins=20, color='blue', edgecolor='black', alpha=0.7)
        ax5.set_title('Trip Count Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Number of Trips', fontsize=10)
        ax5.set_ylabel('Number of Routes', fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Total Distance Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(route_metrics['total_distance'], bins=20, color='orange', edgecolor='black', alpha=0.7)
        ax6.set_title('Total Distance Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Total Distance (miles)', fontsize=10)
        ax6.set_ylabel('Number of Routes', fontsize=10)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Profit Distribution
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.hist(route_metrics['profit'], bins=20, color='purple', edgecolor='black', alpha=0.7)
        ax7.set_title('Profit Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Profit ($)', fontsize=10)
        ax7.set_ylabel('Number of Routes', fontsize=10)
        ax7.grid(axis='y', alpha=0.3)
        
        # 8. Revenue Distribution
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.hist(route_metrics['total_revenue'], bins=20, color='teal', edgecolor='black', alpha=0.7)
        ax8.set_title('Revenue Distribution', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Total Revenue ($)', fontsize=10)
        ax8.set_ylabel('Number of Routes', fontsize=10)
        ax8.grid(axis='y', alpha=0.3)
        
        plt.suptitle('ROUTE PROFITABILITY DASHBOARD', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('Docs/Route_Dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Route dashboard saved to Docs/Route_Dashboard.png")
        return fig
    
    @staticmethod
    def create_kpi_summary_table(driver_metrics, fleet_metrics, route_metrics, fuel_efficiency):
        """Create KPI summary visualization"""
        
        print("Creating KPI summary table...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('KEY PERFORMANCE INDICATORS (KPI) SUMMARY', fontsize=16, fontweight='bold')
        
        # Driver KPIs
        ax1 = axes[0, 0]
        ax1.axis('off')
        driver_kpis = [
            ['Metric', 'Value'],
            ['Total Drivers', f"{len(driver_metrics)}"],
            ['Total Revenue', f"${driver_metrics['total_revenue'].sum():,.0f}"],
            ['Avg Revenue/Driver', f"${driver_metrics['total_revenue'].mean():,.0f}"],
            ['Avg MPG', f"{driver_metrics['avg_mpg'].mean():.2f}"],
            ['Total Miles', f"{driver_metrics['total_miles'].sum():,.0f}"],
            ['On-Time Rate', f"{driver_metrics['on_time_delivery_rate'].mean()*100:.1f}%"],
        ]
        table1 = ax1.table(cellText=driver_kpis, cellLoc='center', loc='center',
                          colWidths=[0.5, 0.5])
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)
        for i in range(len(driver_kpis)):
            table1[(i, 0)].set_facecolor('#E8E8E8')
            if i == 0:
                table1[(i, 1)].set_facecolor('#4472C4')
                table1[(i, 0)].set_facecolor('#4472C4')
                table1[(i, 1)].set_text_props(weight='bold', color='white')
                table1[(i, 0)].set_text_props(weight='bold', color='white')
        ax1.set_title('DRIVER METRICS', fontsize=12, fontweight='bold', pad=20)
        
        # Fleet KPIs
        ax2 = axes[0, 1]
        ax2.axis('off')
        fleet_kpis = [
            ['Metric', 'Value'],
            ['Total Trucks', f"{len(fleet_metrics)}"],
            ['Active Trucks', f"{len(fleet_metrics[fleet_metrics['status'] == 'Active'])}"],
            ['Total Revenue', f"${fleet_metrics['total_revenue'].sum():,.0f}"],
            ['Avg Revenue/Truck', f"${fleet_metrics['total_revenue'].mean():,.0f}"],
            ['Avg Cost/Mile', f"${fleet_metrics['cost_per_mile'].mean():.2f}"],
            ['Total Trips', f"{fleet_metrics['trips_count'].sum():.0f}"],
        ]
        table2 = ax2.table(cellText=fleet_kpis, cellLoc='center', loc='center',
                          colWidths=[0.5, 0.5])
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 2)
        for i in range(len(fleet_kpis)):
            table2[(i, 0)].set_facecolor('#E8E8E8')
            if i == 0:
                table2[(i, 1)].set_facecolor('#70AD47')
                table2[(i, 0)].set_facecolor('#70AD47')
                table2[(i, 1)].set_text_props(weight='bold', color='white')
                table2[(i, 0)].set_text_props(weight='bold', color='white')
        ax2.set_title('FLEET METRICS', fontsize=12, fontweight='bold', pad=20)
        
        # Route KPIs
        ax3 = axes[1, 0]
        ax3.axis('off')
        route_kpis = [
            ['Metric', 'Value'],
            ['Total Routes', f"{len(route_metrics)}"],
            ['Total Profit', f"${route_metrics['profit'].sum():,.0f}"],
            ['Avg Profit/Route', f"${route_metrics['profit'].mean():,.0f}"],
            ['Avg Margin', f"{route_metrics['profit_margin'].mean():.1f}%"],
            ['Total Revenue', f"${route_metrics['total_revenue'].sum():,.0f}"],
            ['Unprofitable Routes', f"{len(route_metrics[route_metrics['profit'] < 0])}"],
        ]
        table3 = ax3.table(cellText=route_kpis, cellLoc='center', loc='center',
                          colWidths=[0.5, 0.5])
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1, 2)
        for i in range(len(route_kpis)):
            table3[(i, 0)].set_facecolor('#E8E8E8')
            if i == 0:
                table3[(i, 1)].set_facecolor('#FFC000')
                table3[(i, 0)].set_facecolor('#FFC000')
                table3[(i, 1)].set_text_props(weight='bold')
                table3[(i, 0)].set_text_props(weight='bold')
        ax3.set_title('ROUTE METRICS', fontsize=12, fontweight='bold', pad=20)
        
        # Fuel Efficiency KPIs
        ax4 = axes[1, 1]
        ax4.axis('off')
        fuel_kpis = [
            ['Metric', 'Value'],
            ['Total Trucks Tracked', f"{len(fuel_efficiency)}"],
            ['Avg Fleet MPG', f"{fuel_efficiency['calculated_mpg'].mean():.2f}"],
            ['Best MPG', f"{fuel_efficiency['calculated_mpg'].max():.2f}"],
            ['Worst MPG', f"{fuel_efficiency['calculated_mpg'].min():.2f}"],
            ['Total Fuel Cost', f"${fuel_efficiency['fuel_cost'].sum():,.0f}"],
            ['Avg Cost/Mile', f"${fuel_efficiency['cost_per_mile'].mean():.2f}"],
        ]
        table4 = ax4.table(cellText=fuel_kpis, cellLoc='center', loc='center',
                          colWidths=[0.5, 0.5])
        table4.auto_set_font_size(False)
        table4.set_fontsize(10)
        table4.scale(1, 2)
        for i in range(len(fuel_kpis)):
            table4[(i, 0)].set_facecolor('#E8E8E8')
            if i == 0:
                table4[(i, 1)].set_facecolor('#C00000')
                table4[(i, 0)].set_facecolor('#C00000')
                table4[(i, 1)].set_text_props(weight='bold', color='white')
                table4[(i, 0)].set_text_props(weight='bold', color='white')
        ax4.set_title('FUEL EFFICIENCY METRICS', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('Docs/KPI_Summary_Table.png', dpi=300, bbox_inches='tight')
        print("✓ KPI summary saved to Docs/KPI_Summary_Table.png")
        return fig


if __name__ == "__main__":
    print("Visualization utilities module loaded successfully!")
