"""
Comprehensive Data Preparation for Spotify Tracks Dataset
CRISP-DM Phase 3: Data Preparation

This script implements professional data cleaning, transformation, and feature engineering
following CRISP-DM methodology and data science best practices.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class SpotifyDataPreparator:
    """
    Comprehensive data preparation class for Spotify tracks dataset.
    Implements professional data science practices for CRISP-DM methodology.
    """
    
    def __init__(self, file_path):
        """Initialize the data preparator with dataset path."""
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.df_processed = None
        self.feature_names = []
        self.preprocessing_steps = []
        
    def load_data(self):
        """Load the dataset."""
        print("=" * 60)
        print("CRISP-DM PHASE 3: DATA PREPARATION")
        print("=" * 60)
        
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"[SUCCESS] Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading dataset: {e}")
            return False
    
    def data_cleaning(self):
        """Comprehensive data cleaning process."""
        print("\n" + "=" * 40)
        print("DATA CLEANING")
        print("=" * 40)
        
        self.df_cleaned = self.df.copy()
        cleaning_steps = []
        
        # 1. Handle missing values
        print("\n[STEP 1] Missing Value Analysis:")
        missing_data = self.df_cleaned.isnull().sum()
        if missing_data.sum() > 0:
            print("   - Missing values found:")
            for col, count in missing_data[missing_data > 0].items():
                percentage = (count / len(self.df_cleaned)) * 100
                print(f"     {col}: {count} ({percentage:.2f}%)")
            cleaning_steps.append("Missing value handling")
        else:
            print("   - No missing values found")
        
        # 2. Handle duplicates
        print("\n[STEP 2] Duplicate Analysis:")
        duplicates = self.df_cleaned.duplicated().sum()
        if duplicates > 0:
            print(f"   - Duplicate records found: {duplicates}")
            self.df_cleaned = self.df_cleaned.drop_duplicates()
            print(f"   - Duplicates removed. New shape: {self.df_cleaned.shape}")
            cleaning_steps.append("Duplicate removal")
        else:
            print("   - No duplicate records found")
        
        # 3. Handle empty strings
        print("\n[STEP 3] Empty String Analysis:")
        empty_strings = {}
        for col in self.df_cleaned.select_dtypes(include=['object']).columns:
            empty_count = (self.df_cleaned[col] == '').sum()
            if empty_count > 0:
                empty_strings[col] = empty_count
        
        if empty_strings:
            print("   - Empty strings found:")
            for col, count in empty_strings.items():
                print(f"     {col}: {count}")
                # Replace empty strings with NaN for proper handling
                self.df_cleaned[col] = self.df_cleaned[col].replace('', np.nan)
            cleaning_steps.append("Empty string handling")
        else:
            print("   - No empty strings found")
        
        # 4. Data type validation and correction
        print("\n[STEP 4] Data Type Validation:")
        
        # Ensure popularity is integer and within valid range
        invalid_popularity = (self.df_cleaned['popularity'] < 0) | (self.df_cleaned['popularity'] > 100)
        if invalid_popularity.sum() > 0:
            print(f"   - Invalid popularity values found: {invalid_popularity.sum()}")
            self.df_cleaned.loc[invalid_popularity, 'popularity'] = np.nan
            cleaning_steps.append("Popularity range validation")
        
        # Ensure duration is positive
        invalid_duration = self.df_cleaned['duration_ms'] <= 0
        if invalid_duration.sum() > 0:
            print(f"   - Invalid duration values found: {invalid_duration.sum()}")
            self.df_cleaned.loc[invalid_duration, 'duration_ms'] = np.nan
            cleaning_steps.append("Duration validation")
        
        # Ensure explicit is boolean
        if self.df_cleaned['explicit'].dtype != 'bool':
            print("   - Converting explicit column to boolean")
            self.df_cleaned['explicit'] = self.df_cleaned['explicit'].astype(bool)
            cleaning_steps.append("Boolean conversion")
        
        # 5. Outlier detection and handling
        print("\n[STEP 5] Outlier Analysis:")
        
        # Duration outliers (extreme values)
        Q1_duration = self.df_cleaned['duration_ms'].quantile(0.25)
        Q3_duration = self.df_cleaned['duration_ms'].quantile(0.75)
        IQR_duration = Q3_duration - Q1_duration
        lower_bound = Q1_duration - 1.5 * IQR_duration
        upper_bound = Q3_duration + 1.5 * IQR_duration
        
        duration_outliers = (self.df_cleaned['duration_ms'] < lower_bound) | (self.df_cleaned['duration_ms'] > upper_bound)
        print(f"   - Duration outliers detected: {duration_outliers.sum()}")
        
        # Cap extreme outliers but keep reasonable ones
        extreme_duration = self.df_cleaned['duration_ms'] > 1800000  # 30 minutes
        if extreme_duration.sum() > 0:
            print(f"   - Extreme duration outliers (>30 min): {extreme_duration.sum()}")
            self.df_cleaned.loc[extreme_duration, 'duration_ms'] = 1800000
            cleaning_steps.append("Extreme duration capping")
        
        # 6. Final data quality check
        print("\n[STEP 6] Final Data Quality Check:")
        final_missing = self.df_cleaned.isnull().sum()
        if final_missing.sum() > 0:
            print("   - Remaining missing values:")
            for col, count in final_missing[final_missing > 0].items():
                percentage = (count / len(self.df_cleaned)) * 100
                print(f"     {col}: {count} ({percentage:.2f}%)")
        else:
            print("   - No missing values remaining")
        
        self.preprocessing_steps.extend(cleaning_steps)
        print(f"\n[SUMMARY] Data cleaning completed. Steps: {', '.join(cleaning_steps)}")
        
        return self.df_cleaned
    
    def feature_engineering(self):
        """Comprehensive feature engineering process."""
        print("\n" + "=" * 40)
        print("FEATURE ENGINEERING")
        print("=" * 40)
        
        self.df_processed = self.df_cleaned.copy()
        feature_steps = []
        
        # 1. Derived numerical features
        print("\n[STEP 1] Numerical Feature Engineering:")
        
        # Duration in minutes
        self.df_processed['duration_minutes'] = self.df_processed['duration_ms'] / 60000
        print("   - Created: duration_minutes")
        
        # Duration categories
        self.df_processed['duration_category'] = pd.cut(
            self.df_processed['duration_minutes'],
            bins=[0, 2, 4, 6, 8, float('inf')],
            labels=['short', 'medium', 'long', 'very_long', 'extended']
        )
        print("   - Created: duration_category")
        
        # Popularity categories
        self.df_processed['popularity_category'] = pd.cut(
            self.df_processed['popularity'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        print("   - Created: popularity_category")
        
        # 2. Text feature engineering
        print("\n[STEP 2] Text Feature Engineering:")
        
        # Track name length
        self.df_processed['name_length'] = self.df_processed['name'].str.len()
        print("   - Created: name_length")
        
        # Artist count (number of artists)
        self.df_processed['artist_count'] = self.df_processed['artists'].str.count(',') + 1
        print("   - Created: artist_count")
        
        # Album name length
        self.df_processed['album_length'] = self.df_processed['album'].str.len()
        print("   - Created: album_length")
        
        # 3. Genre feature engineering
        print("\n[STEP 3] Genre Feature Engineering:")
        
        # Genre frequency (how common is this genre)
        genre_counts = self.df_processed['genre'].value_counts()
        self.df_processed['genre_frequency'] = self.df_processed['genre'].map(genre_counts)
        print("   - Created: genre_frequency")
        
        # Genre popularity (average popularity by genre)
        genre_popularity = self.df_processed.groupby('genre')['popularity'].mean()
        self.df_processed['genre_avg_popularity'] = self.df_processed['genre'].map(genre_popularity)
        print("   - Created: genre_avg_popularity")
        
        # 4. Artist feature engineering
        print("\n[STEP 4] Artist Feature Engineering:")
        
        # Artist frequency (how many tracks per artist)
        artist_counts = self.df_processed['artists'].value_counts()
        self.df_processed['artist_frequency'] = self.df_processed['artists'].map(artist_counts)
        print("   - Created: artist_frequency")
        
        # Artist popularity (average popularity by artist)
        artist_popularity = self.df_processed.groupby('artists')['popularity'].mean()
        self.df_processed['artist_avg_popularity'] = self.df_processed['artists'].map(artist_popularity)
        print("   - Created: artist_avg_popularity")
        
        # 5. Interaction features
        print("\n[STEP 5] Interaction Feature Engineering:")
        
        # Duration * Genre interaction
        self.df_processed['duration_genre_interaction'] = (
            self.df_processed['duration_minutes'] * self.df_processed['genre_avg_popularity']
        )
        print("   - Created: duration_genre_interaction")
        
        # Explicit * Genre interaction
        self.df_processed['explicit_genre_interaction'] = (
            self.df_processed['explicit'].astype(int) * self.df_processed['genre_avg_popularity']
        )
        print("   - Created: explicit_genre_interaction")
        
        feature_steps.extend([
            "Numerical feature engineering",
            "Text feature engineering", 
            "Genre feature engineering",
            "Artist feature engineering",
            "Interaction feature engineering"
        ])
        
        self.preprocessing_steps.extend(feature_steps)
        print(f"\n[SUMMARY] Feature engineering completed. Steps: {len(feature_steps)}")
        
        return self.df_processed
    
    def data_encoding(self):
        """Comprehensive data encoding for machine learning."""
        print("\n" + "=" * 40)
        print("DATA ENCODING")
        print("=" * 40)
        
        encoding_steps = []
        
        # 1. Categorical encoding
        print("\n[STEP 1] Categorical Encoding:")
        
        # Label encoding for ordinal categories
        le_duration = LabelEncoder()
        self.df_processed['duration_category_encoded'] = le_duration.fit_transform(
            self.df_processed['duration_category'].astype(str)
        )
        print("   - Label encoded: duration_category")
        
        le_popularity = LabelEncoder()
        self.df_processed['popularity_category_encoded'] = le_popularity.fit_transform(
            self.df_processed['popularity_category'].astype(str)
        )
        print("   - Label encoded: popularity_category")
        
        # One-hot encoding for genre (top 20 most frequent genres)
        top_genres = self.df_processed['genre'].value_counts().head(20).index
        for genre in top_genres:
            self.df_processed[f'genre_{genre}'] = (self.df_processed['genre'] == genre).astype(int)
        print(f"   - One-hot encoded: top 20 genres")
        
        # 2. Text encoding
        print("\n[STEP 2] Text Encoding:")
        
        # TF-IDF for track names (limited features)
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        name_tfidf = tfidf.fit_transform(self.df_processed['name'].fillna(''))
        name_tfidf_df = pd.DataFrame(
            name_tfidf.toarray(),
            columns=[f'name_tfidf_{i}' for i in range(name_tfidf.shape[1])]
        )
        self.df_processed = pd.concat([self.df_processed, name_tfidf_df], axis=1)
        print("   - TF-IDF encoded: track names (50 features)")
        
        # 3. Numerical scaling
        print("\n[STEP 3] Numerical Scaling:")
        
        # Features to scale
        numerical_features = [
            'duration_minutes', 'name_length', 'artist_count', 'album_length',
            'genre_frequency', 'genre_avg_popularity', 'artist_frequency',
            'artist_avg_popularity', 'duration_genre_interaction',
            'explicit_genre_interaction'
        ]
        
        # Add TF-IDF features
        numerical_features.extend([f'name_tfidf_{i}' for i in range(50)])
        
        # Scale features
        scaler = StandardScaler()
        self.df_processed[numerical_features] = scaler.fit_transform(
            self.df_processed[numerical_features]
        )
        print(f"   - Scaled: {len(numerical_features)} numerical features")
        
        encoding_steps.extend([
            "Categorical encoding",
            "Text encoding",
            "Numerical scaling"
        ])
        
        self.preprocessing_steps.extend(encoding_steps)
        print(f"\n[SUMMARY] Data encoding completed. Steps: {len(encoding_steps)}")
        
        return self.df_processed
    
    def feature_selection(self):
        """Feature selection for modeling."""
        print("\n" + "=" * 40)
        print("FEATURE SELECTION")
        print("=" * 40)
        
        # Define feature sets for different modeling tasks
        print("\n[STEP 1] Feature Set Definition:")
        
        # Base features
        base_features = [
            'duration_minutes', 'explicit', 'name_length', 'artist_count',
            'album_length', 'genre_frequency', 'genre_avg_popularity',
            'artist_frequency', 'artist_avg_popularity'
        ]
        
        # Categorical features
        categorical_features = [
            'duration_category_encoded', 'popularity_category_encoded'
        ]
        
        # Genre features (one-hot encoded)
        genre_features = [col for col in self.df_processed.columns if col.startswith('genre_')]
        
        # TF-IDF features
        tfidf_features = [col for col in self.df_processed.columns if col.startswith('name_tfidf_')]
        
        # Interaction features
        interaction_features = [
            'duration_genre_interaction', 'explicit_genre_interaction'
        ]
        
        # Combine all features
        self.feature_names = (
            base_features + categorical_features + genre_features + 
            tfidf_features + interaction_features
        )
        
        print(f"   - Base features: {len(base_features)}")
        print(f"   - Categorical features: {len(categorical_features)}")
        print(f"   - Genre features: {len(genre_features)}")
        print(f"   - TF-IDF features: {len(tfidf_features)}")
        print(f"   - Interaction features: {len(interaction_features)}")
        print(f"   - Total features: {len(self.feature_names)}")
        
        # Check feature availability
        available_features = [f for f in self.feature_names if f in self.df_processed.columns]
        missing_features = [f for f in self.feature_names if f not in self.df_processed.columns]
        
        if missing_features:
            print(f"\n[WARNING] Missing features: {missing_features}")
            self.feature_names = available_features
        
        print(f"\n[SUMMARY] Feature selection completed. Available features: {len(self.feature_names)}")
        
        return self.feature_names
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """Create train-test split for modeling."""
        print("\n" + "=" * 40)
        print("TRAIN-TEST SPLIT")
        print("=" * 40)
        
        # Prepare features and target
        X = self.df_processed[self.feature_names]
        y = self.df_processed['popularity']
        
        # Create train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"   - Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"   - Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"   - Target variable: popularity (range: {y.min()} - {y.max()})")
        
        return X_train, X_test, y_train, y_test
    
    def generate_preparation_report(self):
        """Generate comprehensive data preparation report."""
        print("\n" + "=" * 60)
        print("DATA PREPARATION SUMMARY REPORT")
        print("=" * 60)
        
        print(f"\n[ORIGINAL DATASET]")
        print(f"   - Shape: {self.df.shape}")
        print(f"   - Features: {list(self.df.columns)}")
        
        print(f"\n[CLEANED DATASET]")
        print(f"   - Shape: {self.df_cleaned.shape}")
        print(f"   - Data loss: {self.df.shape[0] - self.df_cleaned.shape[0]} rows")
        
        print(f"\n[PROCESSED DATASET]")
        print(f"   - Shape: {self.df_processed.shape}")
        print(f"   - New features: {self.df_processed.shape[1] - self.df_cleaned.shape[1]}")
        
        print(f"\n[PREPROCESSING STEPS]")
        for i, step in enumerate(self.preprocessing_steps, 1):
            print(f"   {i}. {step}")
        
        print(f"\n[FEATURE ENGINEERING]")
        print(f"   - Total features for modeling: {len(self.feature_names)}")
        print(f"   - Feature categories:")
        print(f"     * Numerical: {len([f for f in self.feature_names if 'tfidf' not in f and 'genre_' not in f])}")
        print(f"     * Categorical: {len([f for f in self.feature_names if f.startswith('genre_') or 'category' in f])}")
        print(f"     * Text: {len([f for f in self.feature_names if 'tfidf' in f])}")
        
        print(f"\n[DATA QUALITY METRICS]")
        print(f"   - Missing values: {self.df_processed.isnull().sum().sum()}")
        print(f"   - Duplicate rows: {self.df_processed.duplicated().sum()}")
        print(f"   - Data types: {dict(self.df_processed.dtypes.value_counts())}")
        
        print(f"\n[RECOMMENDATIONS FOR MODELING]")
        print("   - Dataset is ready for machine learning")
        print("   - Consider feature importance analysis")
        print("   - Monitor for overfitting with high-dimensional features")
        print("   - Use cross-validation for robust evaluation")
        print("   - Consider dimensionality reduction if needed")
        
        print("\n" + "=" * 60)
        print("END OF DATA PREPARATION PHASE")
        print("=" * 60)
    
    def run_complete_preparation(self):
        """Execute complete data preparation pipeline."""
        if not self.load_data():
            return False
        
        self.data_cleaning()
        self.feature_engineering()
        self.data_encoding()
        self.feature_selection()
        self.generate_preparation_report()
        
        return True

# Main execution
if __name__ == "__main__":
    # Initialize the data preparator
    preparator = SpotifyDataPreparator('spotify_tracks.csv')
    
    # Run complete preparation
    success = preparator.run_complete_preparation()
    
    if success:
        print("\n[SUCCESS] Data Preparation phase completed successfully!")
        print("[INFO] Dataset is ready for modeling phase")
        print("[INFO] Use preparator.df_processed for modeling")
        print("[INFO] Use preparator.feature_names for feature selection")
        
        # Create train-test split for modeling
        X_train, X_test, y_train, y_test = preparator.train_test_split()
        print(f"\n[INFO] Train-test split created:")
        print(f"   - Training: {X_train.shape}")
        print(f"   - Testing: {X_test.shape}")
    else:
        print("\n[ERROR] Data Preparation phase failed!")
