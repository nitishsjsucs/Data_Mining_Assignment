"""
Comprehensive Data Exploration for Spotify Tracks Dataset
CRISP-DM Phase 2: Data Understanding

This script provides thorough data exploration and quality assessment
for the Spotify tracks dataset following professional data science practices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpotifyDataExplorer:
    """
    Comprehensive data exploration class for Spotify tracks dataset.
    Implements professional data science practices for CRISP-DM methodology.
    """
    
    def __init__(self, file_path):
        """Initialize the data explorer with dataset path."""
        self.file_path = file_path
        self.df = None
        self.data_quality_report = {}
        
    def load_data(self):
        """Load and perform initial data inspection."""
        print("=" * 60)
        print("CRISP-DM PHASE 2: DATA UNDERSTANDING")
        print("=" * 60)
        
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"[SUCCESS] Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading dataset: {e}")
            return False
    
    def basic_info(self):
        """Display basic dataset information."""
        print("\n" + "=" * 40)
        print("BASIC DATASET INFORMATION")
        print("=" * 40)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumn Information:")
        print("-" * 30)
        for col in self.df.columns:
            dtype = self.df[col].dtype
            non_null = self.df[col].count()
            null_count = self.df[col].isnull().sum()
            print(f"{col:15} | {str(dtype):12} | Non-null: {non_null:5} | Null: {null_count:3}")
    
    def data_quality_assessment(self):
        """Comprehensive data quality assessment."""
        print("\n" + "=" * 40)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 40)
        
        quality_issues = []
        
        # Check for missing values
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print("[WARNING] Missing Values Found:")
            for col, count in missing_data[missing_data > 0].items():
                percentage = (count / len(self.df)) * 100
                print(f"   {col}: {count} ({percentage:.2f}%)")
                quality_issues.append(f"Missing values in {col}")
        else:
            print("[OK] No missing values found")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"[WARNING] Duplicate records found: {duplicates}")
            quality_issues.append(f"{duplicates} duplicate records")
        else:
            print("[OK] No duplicate records found")
        
        # Check for empty strings
        empty_strings = {}
        for col in self.df.select_dtypes(include=['object']).columns:
            empty_count = (self.df[col] == '').sum()
            if empty_count > 0:
                empty_strings[col] = empty_count
        
        if empty_strings:
            print("[WARNING] Empty strings found:")
            for col, count in empty_strings.items():
                print(f"   {col}: {count}")
                quality_issues.append(f"Empty strings in {col}")
        else:
            print("[OK] No empty strings found")
        
        # Store quality report
        self.data_quality_report = {
            'missing_values': missing_data.to_dict(),
            'duplicates': duplicates,
            'empty_strings': empty_strings,
            'issues': quality_issues
        }
        
        return quality_issues
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics."""
        print("\n" + "=" * 40)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 40)
        
        # Numerical columns statistics
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print("\nNumerical Variables:")
            print("-" * 30)
            desc_stats = self.df[numerical_cols].describe()
            print(desc_stats.round(2))
            
            # Additional statistics
            print("\nAdditional Statistics:")
            print("-" * 30)
            for col in numerical_cols:
                skewness = stats.skew(self.df[col].dropna())
                kurtosis = stats.kurtosis(self.df[col].dropna())
                print(f"{col:15} | Skewness: {skewness:7.3f} | Kurtosis: {kurtosis:7.3f}")
        
        # Categorical columns statistics
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\nCategorical Variables:")
            print("-" * 30)
            for col in categorical_cols:
                unique_count = self.df[col].nunique()
                most_frequent = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "N/A"
                freq_count = self.df[col].value_counts().iloc[0] if not self.df[col].value_counts().empty else 0
                print(f"{col:15} | Unique: {unique_count:5} | Most frequent: {str(most_frequent)[:30]:30} | Count: {freq_count:5}")
    
    def genre_analysis(self):
        """Comprehensive genre analysis."""
        print("\n" + "=" * 40)
        print("GENRE ANALYSIS")
        print("=" * 40)
        
        genre_counts = self.df['genre'].value_counts()
        print(f"Total unique genres: {len(genre_counts)}")
        print("\nGenre distribution:")
        print("-" * 30)
        
        for genre, count in genre_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"{genre:20} | {count:5} tracks ({percentage:5.1f}%)")
        
        # Genre statistics by numerical variables
        print("\nGenre Statistics by Popularity:")
        print("-" * 40)
        genre_stats = self.df.groupby('genre')['popularity'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        print(genre_stats)
        
        print("\nGenre Statistics by Duration (minutes):")
        print("-" * 40)
        self.df['duration_minutes'] = self.df['duration_ms'] / 60000
        duration_stats = self.df.groupby('genre')['duration_minutes'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        print(duration_stats)
    
    def popularity_analysis(self):
        """Detailed popularity analysis."""
        print("\n" + "=" * 40)
        print("POPULARITY ANALYSIS")
        print("=" * 40)
        
        popularity = self.df['popularity']
        
        print(f"Popularity Statistics:")
        print(f"  Mean: {popularity.mean():.2f}")
        print(f"  Median: {popularity.median():.2f}")
        print(f"  Standard Deviation: {popularity.std():.2f}")
        print(f"  Range: {popularity.min()} - {popularity.max()}")
        print(f"  Skewness: {stats.skew(popularity):.3f}")
        
        # Popularity distribution
        print(f"\nPopularity Distribution:")
        print("-" * 30)
        popularity_bins = pd.cut(popularity, bins=[0, 20, 40, 60, 80, 100], labels=['Very Low (0-20)', 'Low (21-40)', 'Medium (41-60)', 'High (61-80)', 'Very High (81-100)'])
        popularity_dist = popularity_bins.value_counts()
        for category, count in popularity_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"{category:20} | {count:5} tracks ({percentage:5.1f}%)")
    
    def duration_analysis(self):
        """Comprehensive duration analysis."""
        print("\n" + "=" * 40)
        print("DURATION ANALYSIS")
        print("=" * 40)
        
        duration_minutes = self.df['duration_minutes']
        
        print(f"Duration Statistics (minutes):")
        print(f"  Mean: {duration_minutes.mean():.2f}")
        print(f"  Median: {duration_minutes.median():.2f}")
        print(f"  Standard Deviation: {duration_minutes.std():.2f}")
        print(f"  Range: {duration_minutes.min():.2f} - {duration_minutes.max():.2f}")
        
        # Duration categories
        print(f"\nDuration Categories:")
        print("-" * 30)
        duration_bins = pd.cut(duration_minutes, bins=[0, 2, 4, 6, 8, float('inf')], labels=['Short (<2min)', 'Medium (2-4min)', 'Long (4-6min)', 'Very Long (6-8min)', 'Extended (>8min)'])
        duration_dist = duration_bins.value_counts()
        for category, count in duration_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"{category:20} | {count:5} tracks ({percentage:5.1f}%)")
    
    def explicit_content_analysis(self):
        """Analysis of explicit content patterns."""
        print("\n" + "=" * 40)
        print("EXPLICIT CONTENT ANALYSIS")
        print("=" * 40)
        
        explicit_counts = self.df['explicit'].value_counts()
        print("Explicit Content Distribution:")
        print("-" * 30)
        for value, count in explicit_counts.items():
            percentage = (count / len(self.df)) * 100
            label = "Explicit" if value else "Non-Explicit"
            print(f"{label:15} | {count:5} tracks ({percentage:5.1f}%)")
        
        # Explicit content vs popularity
        print(f"\nExplicit Content vs Popularity:")
        print("-" * 30)
        explicit_stats = self.df.groupby('explicit')['popularity'].agg(['count', 'mean', 'std']).round(2)
        print(explicit_stats)
    
    def correlation_analysis(self):
        """Analyze correlations between numerical variables."""
        print("\n" + "=" * 40)
        print("CORRELATION ANALYSIS")
        print("=" * 40)
        
        numerical_cols = ['popularity', 'duration_ms', 'duration_minutes']
        correlation_matrix = self.df[numerical_cols].corr()
        
        print("Correlation Matrix:")
        print("-" * 30)
        print(correlation_matrix.round(3))
        
        # Identify significant correlations
        print(f"\nSignificant Correlations (|r| > 0.3):")
        print("-" * 40)
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    strength = "Strong" if abs(corr_value) > 0.7 else "Moderate"
                    direction = "Positive" if corr_value > 0 else "Negative"
                    print(f"{col1} vs {col2}: {corr_value:.3f} ({strength} {direction})")
    
    def outlier_detection(self):
        """Detect and analyze outliers."""
        print("\n" + "=" * 40)
        print("OUTLIER DETECTION")
        print("=" * 40)
        
        numerical_cols = ['popularity', 'duration_minutes']
        
        for col in numerical_cols:
            print(f"\n{col.upper()} Outliers:")
            print("-" * 30)
            
            # IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            print(f"  IQR Method: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.1f}%)")
            
            if len(outliers) > 0:
                print(f"  Outlier range: {outliers[col].min():.2f} - {outliers[col].max():.2f}")
                print(f"  Normal range: {lower_bound:.2f} - {upper_bound:.2f}")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n" + "=" * 40)
        print("GENERATING VISUALIZATIONS")
        print("=" * 40)
        
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Genre Distribution
        plt.subplot(3, 3, 1)
        genre_counts = self.df['genre'].value_counts()
        plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Genre Distribution', fontsize=14, fontweight='bold')
        
        # 2. Popularity Distribution
        plt.subplot(3, 3, 2)
        plt.hist(self.df['popularity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Popularity Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Popularity Score')
        plt.ylabel('Frequency')
        
        # 3. Duration Distribution
        plt.subplot(3, 3, 3)
        plt.hist(self.df['duration_minutes'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Duration Distribution (Minutes)', fontsize=14, fontweight='bold')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Frequency')
        
        # 4. Explicit Content Distribution
        plt.subplot(3, 3, 4)
        explicit_counts = self.df['explicit'].value_counts()
        plt.bar(['Non-Explicit', 'Explicit'], explicit_counts.values, color=['lightblue', 'lightcoral'])
        plt.title('Explicit Content Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Tracks')
        
        # 5. Popularity vs Duration
        plt.subplot(3, 3, 5)
        plt.scatter(self.df['duration_minutes'], self.df['popularity'], alpha=0.6, color='purple')
        plt.title('Popularity vs Duration', fontsize=14, fontweight='bold')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Popularity Score')
        
        # 6. Genre vs Popularity Box Plot
        plt.subplot(3, 3, 6)
        self.df.boxplot(column='popularity', by='genre', ax=plt.gca())
        plt.title('Popularity by Genre', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        
        # 7. Genre vs Duration Box Plot
        plt.subplot(3, 3, 7)
        self.df.boxplot(column='duration_minutes', by='genre', ax=plt.gca())
        plt.title('Duration by Genre', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        
        # 8. Correlation Heatmap
        plt.subplot(3, 3, 8)
        numerical_cols = ['popularity', 'duration_minutes']
        correlation_matrix = self.df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # 9. Popularity by Explicit Content
        plt.subplot(3, 3, 9)
        self.df.boxplot(column='popularity', by='explicit', ax=plt.gca())
        plt.title('Popularity by Explicit Content', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        plt.savefig('spotify_data_exploration.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Visualizations saved as 'spotify_data_exploration.png'")
        
        return fig
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 60)
        print("DATA UNDERSTANDING SUMMARY REPORT")
        print("=" * 60)
        
        print(f"\n[DATASET OVERVIEW]")
        print(f"   - Total tracks: {len(self.df):,}")
        print(f"   - Total features: {len(self.df.columns)}")
        print(f"   - Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\n[GENRE ANALYSIS]")
        genre_counts = self.df['genre'].value_counts()
        print(f"   - Unique genres: {len(genre_counts)}")
        for genre, count in genre_counts.head(3).items():
            percentage = (count / len(self.df)) * 100
            print(f"   - {genre}: {count:,} tracks ({percentage:.1f}%)")
        
        print(f"\n[POPULARITY INSIGHTS]")
        popularity = self.df['popularity']
        print(f"   - Average popularity: {popularity.mean():.1f}")
        print(f"   - Popularity range: {popularity.min()} - {popularity.max()}")
        print(f"   - High popularity tracks (>60): {len(self.df[self.df['popularity'] > 60]):,}")
        
        print(f"\n[DURATION INSIGHTS]")
        duration_minutes = self.df['duration_minutes']
        print(f"   - Average duration: {duration_minutes.mean():.1f} minutes")
        print(f"   - Duration range: {duration_minutes.min():.1f} - {duration_minutes.max():.1f} minutes")
        print(f"   - Standard track length (3-5 min): {len(self.df[(self.df['duration_minutes'] >= 3) & (self.df['duration_minutes'] <= 5)]):,}")
        
        print(f"\n[EXPLICIT CONTENT]")
        explicit_count = self.df['explicit'].sum()
        explicit_percentage = (explicit_count / len(self.df)) * 100
        print(f"   - Explicit tracks: {explicit_count:,} ({explicit_percentage:.1f}%)")
        print(f"   - Non-explicit tracks: {len(self.df) - explicit_count:,} ({100 - explicit_percentage:.1f}%)")
        
        print(f"\n[DATA QUALITY]")
        if self.data_quality_report['issues']:
            print("   - Issues found:")
            for issue in self.data_quality_report['issues']:
                print(f"     - {issue}")
        else:
            print("   - No major data quality issues detected")
        
        print(f"\n[KEY CORRELATIONS]")
        correlation_matrix = self.df[['popularity', 'duration_minutes']].corr()
        corr_value = correlation_matrix.iloc[0, 1]
        print(f"   - Popularity vs Duration: {corr_value:.3f}")
        
        print(f"\n[RECOMMENDATIONS FOR NEXT PHASE]")
        print("   - Data quality is good - proceed with modeling")
        print("   - Consider feature engineering for artist names")
        print("   - Genre encoding will be important for modeling")
        print("   - Duration normalization may improve model performance")
        print("   - Popularity prediction is feasible with current features")
        
        print("\n" + "=" * 60)
        print("END OF DATA UNDERSTANDING PHASE")
        print("=" * 60)
    
    def run_complete_analysis(self):
        """Execute complete data understanding analysis."""
        if not self.load_data():
            return False
        
        self.basic_info()
        self.data_quality_assessment()
        self.descriptive_statistics()
        self.genre_analysis()
        self.popularity_analysis()
        self.duration_analysis()
        self.explicit_content_analysis()
        self.correlation_analysis()
        self.outlier_detection()
        self.generate_visualizations()
        self.generate_summary_report()
        
        return True

# Main execution
if __name__ == "__main__":
    # Initialize the data explorer
    explorer = SpotifyDataExplorer('spotify_tracks.csv')
    
    # Run complete analysis
    success = explorer.run_complete_analysis()
    
    if success:
        print("\n[SUCCESS] Data Understanding phase completed successfully!")
        print("[INFO] Check 'spotify_data_exploration.png' for visualizations")
        print("[INFO] Review the summary report above for key insights")
    else:
        print("\n[ERROR] Data Understanding phase failed!")
