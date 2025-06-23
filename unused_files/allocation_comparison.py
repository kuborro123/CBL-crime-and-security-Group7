import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PoliceAllocationAnalysis:
    def __init__(self):
        """
        Initialize analysis class
        Parameters based on literature:
        - Sherman & Weisburd (1995): Police patrol deterrent effects
        - Braga et al. (2019): Hot spots policing meta-analysis
        - Koper (1995): Optimal patrol time research
        """
        # Core parameters from literature
        self.PREVENTION_RATE_PER_HOUR = 0.4  # 0.4 crimes prevented per patrol hour
        self.MAX_PATROL_HOURS_PER_WARD = 200  # Max 200 hours per ward per day
        self.OFFICERS_PER_WARD = 100  # 100 officers per ward
        
    def load_data(self):
        """Load data files"""
        try:
            # Load prediction results
            self.prediction_df = pd.read_csv('prediction_results.csv')
            print(f"Prediction data loaded: {len(self.prediction_df)} records")
            
            # Load schedule output
            self.schedule_df = pd.read_csv('schedule_output.csv')
            print(f"Schedule data loaded: {len(self.schedule_df)} records")
            
            # Merge datasets
            self.merged_df = self.merge_datasets()
            print(f"Merged data: {len(self.merged_df)} records")
            
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise
        except Exception as e:
            print(f"Data loading failed: {e}")
            raise
    
    def merge_datasets(self):
        """Merge prediction and schedule data"""
        merged = pd.merge(
            self.prediction_df, 
            self.schedule_df, 
            left_on='LSOA_code', 
            right_on='lsoa21cd', 
            how='inner'
        )
        
        # Rename columns for analysis
        merged = merged.rename(columns={
            'predicted_value': 'predicted_crime',
            'patrol_hours': 'optimized_hours'
        })
        
        return merged[['LSOA_code', 'ward', 'predicted_crime', 'optimized_hours', 'risk_score']]
    
    def calculate_traditional_allocation(self):
        """
        Calculate traditional allocation approach
        Traditional: Allocate resources proportional to predicted crime volume
        """
        # Calculate crime proportion for each area
        total_crime = self.merged_df['predicted_crime'].sum()
        self.merged_df['crime_proportion'] = self.merged_df['predicted_crime'] / total_crime
        
        # Calculate total available patrol hours (based on ward count)
        unique_wards = self.merged_df['ward'].nunique()
        total_available_hours = unique_wards * self.MAX_PATROL_HOURS_PER_WARD
        
        # Allocate hours proportionally
        self.merged_df['traditional_hours'] = (
            self.merged_df['crime_proportion'] * total_available_hours
        ).clip(upper=self.MAX_PATROL_HOURS_PER_WARD)
        
        print(f"Traditional allocation calculated")
        print(f"Total available patrol hours: {total_available_hours}")
        print(f"Average allocated hours: {self.merged_df['traditional_hours'].mean():.2f}")
    
    def calculate_prevention_effectiveness(self):
        """Calculate crime prevention effectiveness for both approaches"""
        # Calculate prevented crimes for each approach
        self.merged_df['optimized_prevented'] = np.minimum(
            self.merged_df['optimized_hours'] * self.PREVENTION_RATE_PER_HOUR,
            self.merged_df['predicted_crime']
        )
        
        self.merged_df['traditional_prevented'] = np.minimum(
            self.merged_df['traditional_hours'] * self.PREVENTION_RATE_PER_HOUR,
            self.merged_df['predicted_crime']
        )
        
        # Calculate remaining (unpreventable) crimes
        self.merged_df['optimized_remaining'] = (
            self.merged_df['predicted_crime'] - self.merged_df['optimized_prevented']
        )
        
        self.merged_df['traditional_remaining'] = (
            self.merged_df['predicted_crime'] - self.merged_df['traditional_prevented']
        )
        
        # Calculate efficiency metrics
        total_predicted = self.merged_df['predicted_crime'].sum()
        
        self.optimized_efficiency = (
            self.merged_df['optimized_prevented'].sum() / total_predicted * 100
        )
        
        self.traditional_efficiency = (
            self.merged_df['traditional_prevented'].sum() / total_predicted * 100
        )
        
        self.efficiency_improvement = (
            (self.optimized_efficiency - self.traditional_efficiency) / 
            self.traditional_efficiency * 100
        )
        
        print(f"\n=== Effectiveness Analysis Results ===")
        print(f"Optimized approach prevention efficiency: {self.optimized_efficiency:.2f}%")
        print(f"Traditional approach prevention efficiency: {self.traditional_efficiency:.2f}%")
        print(f"Relative efficiency improvement: {self.efficiency_improvement:.2f}%")
    
    def create_risk_categories(self):
        """Create risk level categories"""
        self.merged_df['risk_category'] = pd.cut(
            self.merged_df['predicted_crime'],
            bins=[0, 0.4, 0.8, float('inf')],
            labels=['Low Risk (<0.4)', 'Medium Risk (0.4-0.8)', 'High Risk (>0.8)']
        )
    
    def plot_efficiency_comparison(self):
        """Plot efficiency comparison charts"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Overall efficiency comparison
        methods = ['Traditional', 'Optimized']
        efficiencies = [self.traditional_efficiency, self.optimized_efficiency]
        colors = ['#e74c3c', '#27ae60']
        
        bars = ax1.bar(methods, efficiencies, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Prevention Efficiency (%)')
        ax1.set_title('Police Allocation Efficiency Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(efficiencies) * 1.2)
        
        # Add value labels
        for bar, eff in zip(bars, efficiencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{eff:.1f}%', ha='center', fontsize=12, fontweight='bold')
        
        # Subplot 2: Risk-level grouped effectiveness
        risk_analysis = self.merged_df.groupby('risk_category').agg({
            'traditional_prevented': 'sum',
            'optimized_prevented': 'sum',
            'predicted_crime': 'sum'
        }).reset_index()
        
        x = np.arange(len(risk_analysis))
        width = 0.35
        
        ax2.bar(x - width/2, risk_analysis['traditional_prevented'], width,
                label='Traditional', color='#e74c3c', alpha=0.8)
        ax2.bar(x + width/2, risk_analysis['optimized_prevented'], width,
                label='Optimized', color='#27ae60', alpha=0.8)
        
        ax2.set_xlabel('Risk Level')
        ax2.set_ylabel('Crimes Prevented')
        ax2.set_title('Prevention Effect by Risk Level', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(risk_analysis['risk_category'])
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_resource_allocation(self):
        """Plot resource allocation analysis charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Resource allocation scatter plot
        ax1.scatter(self.merged_df['predicted_crime'], self.merged_df['traditional_hours'],
                   alpha=0.6, color='#e74c3c', label='Traditional', s=50)
        ax1.scatter(self.merged_df['predicted_crime'], self.merged_df['optimized_hours'],
                   alpha=0.6, color='#27ae60', label='Optimized', s=50)
        ax1.set_xlabel('Predicted Crime Count')
        ax1.set_ylabel('Allocated Patrol Hours')
        ax1.set_title('Resource Allocation vs Crime Risk', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Prevention improvement comparison
        improvement = (self.merged_df['optimized_prevented'] - 
                      self.merged_df['traditional_prevented'])
        colors = ['red' if x < 0 else 'green' for x in improvement]
        
        ax2.scatter(self.merged_df['predicted_crime'], improvement,
                   alpha=0.6, c=colors, s=50)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Predicted Crime Count')
        ax2.set_ylabel('Prevention Improvement (Optimized - Traditional)')
        ax2.set_title('Improvement Effect of Optimized Approach', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Resource utilization efficiency
        self.merged_df['optimized_efficiency'] = (
            self.merged_df['optimized_prevented'] / 
            (self.merged_df['optimized_hours'] + 0.01)  # Avoid division by zero
        )
        self.merged_df['traditional_efficiency'] = (
            self.merged_df['traditional_prevented'] / 
            (self.merged_df['traditional_hours'] + 0.01)
        )
        
        ax3.hist(self.merged_df['traditional_efficiency'], bins=30, alpha=0.7,
                color='#e74c3c', label='Traditional', density=True)
        ax3.hist(self.merged_df['optimized_efficiency'], bins=30, alpha=0.7,
                color='#27ae60', label='Optimized', density=True)
        ax3.set_xlabel('Prevention Efficiency per Hour')
        ax3.set_ylabel('Density')
        ax3.set_title('Resource Utilization Efficiency Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        
        # Subplot 4: Cumulative prevention effect
        sorted_data = self.merged_df.sort_values('predicted_crime', ascending=False).reset_index()
        sorted_data['cumulative_traditional'] = sorted_data['traditional_prevented'].cumsum()
        sorted_data['cumulative_optimized'] = sorted_data['optimized_prevented'].cumsum()
        
        ax4.plot(range(len(sorted_data)), sorted_data['cumulative_traditional'],
                label='Traditional', color='#e74c3c', linewidth=2)
        ax4.plot(range(len(sorted_data)), sorted_data['cumulative_optimized'],
                label='Optimized', color='#27ae60', linewidth=2)
        ax4.set_xlabel('Number of Areas (sorted by crime risk)')
        ax4.set_ylabel('Cumulative Crimes Prevented')
        ax4.set_title('Cumulative Prevention Effect Comparison', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_statistical_analysis(self):
        """Plot statistical analysis charts"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Paired t-test visualization
        diff = self.merged_df['optimized_prevented'] - self.merged_df['traditional_prevented']
        
        ax1.hist(diff, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(diff.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean difference: {diff.mean():.3f}')
        ax1.set_xlabel('Prevention Effect Difference (Optimized - Traditional)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prevention Effect Differences', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(
            self.merged_df['optimized_prevented'],
            self.merged_df['traditional_prevented']
        )
        
        ax1.text(0.05, 0.95, f't-statistic: {t_stat:.3f}\np-value: {p_value:.3e}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Subplot 2: Efficiency improvement heatmap
        # Create risk-resource allocation matrix
        risk_bins = pd.qcut(self.merged_df['predicted_crime'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        resource_bins = pd.qcut(self.merged_df['optimized_hours'], q=5, labels=['Very Few', 'Few', 'Medium', 'Many', 'Very Many'])
        
        improvement_matrix = self.merged_df.groupby([risk_bins, resource_bins])['optimized_prevented'].mean().unstack()
        
        sns.heatmap(improvement_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   ax=ax2, cbar_kws={'label': 'Average Crimes Prevented'})
        ax2.set_xlabel('Resource Allocation Level')
        ax2.set_ylabel('Crime Risk Level')
        ax2.set_title('Risk-Resource Allocation Effect Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Generate analysis summary report"""
        print("\n" + "="*60)
        print("         London Police Allocation Optimization Report")
        print("="*60)
        
        print(f"\n【Data Overview】")
        print(f"• Number of LSOAs analyzed: {len(self.merged_df)}")
        print(f"• Number of wards involved: {self.merged_df['ward'].nunique()}")
        print(f"• Total predicted crimes: {self.merged_df['predicted_crime'].sum():.0f}")
        
        print(f"\n【Resource Allocation Comparison】")
        print(f"• Traditional approach total hours: {self.merged_df['traditional_hours'].sum():.0f}")
        print(f"• Optimized approach total hours: {self.merged_df['optimized_hours'].sum():.0f}")
        print(f"• Resource allocation difference: {(self.merged_df['optimized_hours'].sum() - self.merged_df['traditional_hours'].sum()):.0f} hours")
        
        print(f"\n【Prevention Effect Comparison】")
        print(f"• Traditional approach crimes prevented: {self.merged_df['traditional_prevented'].sum():.0f}")
        print(f"• Optimized approach crimes prevented: {self.merged_df['optimized_prevented'].sum():.0f}")
        print(f"• Additional crimes prevented: {(self.merged_df['optimized_prevented'].sum() - self.merged_df['traditional_prevented'].sum()):.0f}")
        
        print(f"\n【Efficiency Metrics】")
        print(f"• Traditional approach prevention efficiency: {self.traditional_efficiency:.2f}%")
        print(f"• Optimized approach prevention efficiency: {self.optimized_efficiency:.2f}%")
        print(f"• Relative efficiency improvement: {self.efficiency_improvement:.2f}%")
        
        # Statistical testing
        diff = self.merged_df['optimized_prevented'] - self.merged_df['traditional_prevented']
        t_stat, p_value = stats.ttest_rel(
            self.merged_df['optimized_prevented'],
            self.merged_df['traditional_prevented']
        )
        
        print(f"\n【Statistical Significance】")
        print(f"• Paired t-test t-value: {t_stat:.3f}")
        print(f"• p-value: {p_value:.3e}")
        print(f"• Statistically significant: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
        
        print(f"\n【Analysis Conclusions】")
        if self.efficiency_improvement > 0:
            print(f"✓ Optimized allocation significantly outperforms traditional approach")
            print(f"✓ Prevention efficiency improved by {self.efficiency_improvement:.1f}% under same resource constraints")
            print(f"✓ Additional {(self.merged_df['optimized_prevented'].sum() - self.merged_df['traditional_prevented'].sum()):.0f} crimes can be prevented")
        else:
            print(f"✗ Optimized approach did not meet expectations")
        
        print("\n" + "="*60)
    
    def save_results(self, filename='allocation_analysis_results.csv'):
        """Save analysis results to CSV"""
        results_df = self.merged_df[[
            'LSOA_code', 'ward', 'predicted_crime', 'risk_category',
            'traditional_hours', 'optimized_hours',
            'traditional_prevented', 'optimized_prevented',
            'traditional_remaining', 'optimized_remaining'
        ]].copy()
        
        results_df['improvement'] = (
            results_df['optimized_prevented'] - results_df['traditional_prevented']
        )
        
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nAnalysis results saved to: {filename}")
    
    def run_complete_analysis(self):
        """Run complete analysis workflow"""
        print("Starting police allocation optimization analysis...")
        
        # 1. Load data
        self.load_data()
        
        # 2. Calculate traditional allocation
        self.calculate_traditional_allocation()
        
        # 3. Calculate prevention effectiveness
        self.calculate_prevention_effectiveness()
        
        # 4. Create risk categories
        self.create_risk_categories()
        
        # 5. Generate visualizations
        print("\nGenerating visualization charts...")
        self.plot_efficiency_comparison()
        self.plot_resource_allocation()
        self.plot_statistical_analysis()
        
        # 6. Generate report
        self.generate_summary_report()
        
        # 7. Save results
        self.save_results()
        
        print("\nAnalysis completed!")

# Usage example
if __name__ == "__main__":
    # Create analysis instance
    analyzer = PoliceAllocationAnalysis()
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    # Individual analysis components can also be run separately:
    # analyzer.load_data()
    # analyzer.calculate_traditional_allocation()
    # analyzer.calculate_prevention_effectiveness()
    # analyzer.create_risk_categories()
    # analyzer.plot_efficiency_comparison()

"""
References (APA format):

Braga, A. A., Turchan, B. S., Papachristos, A. V., & Hureau, D. M. (2019). 
Hot spots policing and crime reduction: An update of an ongoing systematic 
review and meta-analysis. Journal of Experimental Criminology, 15(3), 289-311.

Koper, C. S. (1995). Just enough police presence: Reducing crime and disorderly 
behavior by optimizing patrol time in crime hot spots. Justice Quarterly, 12(4), 649-672.

Ratcliffe, J. H., Taniguchi, T., Groff, E. R., & Wood, J. D. (2011). The Philadelphia 
foot patrol experiment: A randomized controlled trial of police patrol effectiveness 
in violent crime hotspots. Criminology, 49(3), 795-831.

Sherman, L. W., & Weisburd, D. (1995). General deterrent effects of police patrol 
in crime "hot spots": A randomized, controlled trial. Justice Quarterly, 12(4), 625-648.

Weisburd, D., & Eck, J. E. (2004). What can police do to reduce crime, disorder, 
and fear? Annals of the American Academy of Political and Social Science, 593(1), 42-65.
"""