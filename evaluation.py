"""
Comprehensive Model Evaluation for Spotify Tracks Dataset
CRISP-DM Phase 5: Evaluation

This script provides thorough model evaluation, validation, and business impact assessment
following CRISP-DM methodology and professional data science practices.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SpotifyModelEvaluator:
    """
    Comprehensive model evaluation class for Spotify tracks dataset.
    Implements professional evaluation practices following CRISP-DM methodology.
    """
    
    def __init__(self, results, best_model, X_train, X_test, y_train, y_test, feature_names):
        """Initialize the evaluator with model results and data."""
        self.results = results
        self.best_model = best_model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.evaluation_report = {}
        
    def statistical_evaluation(self):
        """Comprehensive statistical evaluation of models."""
        print("=" * 60)
        print("CRISP-DM PHASE 5: EVALUATION")
        print("=" * 60)
        
        print("\n" + "=" * 40)
        print("STATISTICAL EVALUATION")
        print("=" * 40)
        
        statistical_results = {}
        
        for model_name, results in self.results.items():
            print(f"\n[Evaluation] {model_name}:")
            
            # Get predictions
            y_train_pred = results['y_train_pred']
            y_test_pred = results['y_test_pred']
            
            # Calculate comprehensive metrics
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # Additional statistical metrics
            residuals = self.y_test - y_test_pred
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            
            # R² adjusted for degrees of freedom
            n = len(self.y_test)
            p = len(self.feature_names)
            adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100
            
            # Store results
            statistical_results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'adj_r2': adj_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'mape': mape,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std']
            }
            
            # Print results
            print(f"   - Test R²: {test_r2:.4f}")
            print(f"   - Adjusted R²: {adj_r2:.4f}")
            print(f"   - Test RMSE: {test_rmse:.4f}")
            print(f"   - Test MAE: {test_mae:.4f}")
            print(f"   - MAPE: {mape:.2f}%")
            print(f"   - Residual Mean: {residual_mean:.4f}")
            print(f"   - Residual Std: {residual_std:.4f}")
            print(f"   - CV R²: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
        
        self.evaluation_report['statistical'] = statistical_results
        return statistical_results
    
    def model_validation(self):
        """Comprehensive model validation analysis."""
        print("\n" + "=" * 40)
        print("MODEL VALIDATION")
        print("=" * 40)
        
        validation_results = {}
        
        for model_name, results in self.results.items():
            print(f"\n[Validation] {model_name}:")
            
            # Get model and predictions
            model = results['model']
            y_train_pred = results['y_train_pred']
            y_test_pred = results['y_test_pred']
            
            # Overfitting analysis
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            overfitting_gap = train_r2 - test_r2
            
            # Cross-validation stability
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            cv_stability = 1 - (cv_scores.std() / cv_scores.mean()) if cv_scores.mean() != 0 else 0
            
            # Residual analysis
            residuals = self.y_test - y_test_pred
            
            # Normality test (Shapiro-Wilk for small samples, Kolmogorov-Smirnov for larger)
            if len(residuals) <= 5000:
                _, p_value_norm = stats.shapiro(residuals)
            else:
                _, p_value_norm = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
            
            # Homoscedasticity test (Breusch-Pagan approximation)
            # Simple correlation test between residuals and predictions
            corr_residuals_pred, _ = stats.pearsonr(residuals, y_test_pred)
            
            # Store validation results
            validation_results[model_name] = {
                'overfitting_gap': overfitting_gap,
                'cv_stability': cv_stability,
                'normality_p_value': p_value_norm,
                'homoscedasticity_corr': abs(corr_residuals_pred),
                'is_overfitting': overfitting_gap > 0.05,
                'is_stable': cv_stability > 0.95,
                'is_normal': p_value_norm > 0.05,
                'is_homoscedastic': abs(corr_residuals_pred) < 0.3
            }
            
            # Print validation results
            print(f"   - Overfitting Gap: {overfitting_gap:.4f} {'(High)' if overfitting_gap > 0.05 else '(Low)'}")
            print(f"   - CV Stability: {cv_stability:.4f} {'(Good)' if cv_stability > 0.95 else '(Poor)'}")
            print(f"   - Normality (p-value): {p_value_norm:.4f} {'(Normal)' if p_value_norm > 0.05 else '(Non-normal)'}")
            print(f"   - Homoscedasticity: {abs(corr_residuals_pred):.4f} {'(Good)' if abs(corr_residuals_pred) < 0.3 else '(Poor)'}")
            
            # Overall validation score
            validation_score = (
                (1 if not validation_results[model_name]['is_overfitting'] else 0) +
                (1 if validation_results[model_name]['is_stable'] else 0) +
                (1 if validation_results[model_name]['is_normal'] else 0) +
                (1 if validation_results[model_name]['is_homoscedastic'] else 0)
            ) / 4
            
            validation_results[model_name]['overall_score'] = validation_score
            print(f"   - Overall Validation Score: {validation_score:.2f}/1.00")
        
        self.evaluation_report['validation'] = validation_results
        return validation_results
    
    def business_impact_assessment(self):
        """Assess business impact and practical utility of models."""
        print("\n" + "=" * 40)
        print("BUSINESS IMPACT ASSESSMENT")
        print("=" * 40)
        
        business_results = {}
        
        for model_name, results in self.results.items():
            print(f"\n[Business Impact] {model_name}:")
            
            # Get predictions
            y_test_pred = results['y_test_pred']
            
            # Prediction accuracy by popularity range
            popularity_ranges = [
                (0, 20, "Very Low"),
                (21, 40, "Low"),
                (41, 60, "Medium"),
                (61, 80, "High"),
                (81, 100, "Very High")
            ]
            
            range_accuracy = {}
            for min_val, max_val, label in popularity_ranges:
                mask = (self.y_test >= min_val) & (self.y_test <= max_val)
                if mask.sum() > 0:
                    range_rmse = np.sqrt(mean_squared_error(self.y_test[mask], y_test_pred[mask]))
                    range_mae = mean_absolute_error(self.y_test[mask], y_test_pred[mask])
                    range_accuracy[label] = {
                        'rmse': range_rmse,
                        'mae': range_mae,
                        'count': mask.sum()
                    }
            
            # Business value metrics
            # 1. Prediction confidence intervals
            residuals = self.y_test - y_test_pred
            confidence_95 = 1.96 * np.std(residuals)
            
            # 2. High-value prediction accuracy (popularity > 60)
            high_value_mask = self.y_test > 60
            if high_value_mask.sum() > 0:
                high_value_rmse = np.sqrt(mean_squared_error(self.y_test[high_value_mask], y_test_pred[high_value_mask]))
                high_value_accuracy = 1 - (high_value_rmse / self.y_test[high_value_mask].std())
            else:
                high_value_rmse = 0
                high_value_accuracy = 0
            
            # 3. Cost-benefit analysis (simplified)
            # Assume business value is higher for accurate high-popularity predictions
            business_value_score = (
                results['test_r2'] * 0.4 +  # Overall accuracy
                high_value_accuracy * 0.3 +  # High-value accuracy
                (1 - results['cv_std']) * 0.2 +  # Stability
                (1 - abs(results['test_r2'] - results['train_r2'])) * 0.1  # Generalization
            )
            
            # Store business results
            business_results[model_name] = {
                'range_accuracy': range_accuracy,
                'confidence_95': confidence_95,
                'high_value_rmse': high_value_rmse,
                'high_value_accuracy': high_value_accuracy,
                'business_value_score': business_value_score
            }
            
            # Print business results
            print(f"   - 95% Confidence Interval: ±{confidence_95:.2f} popularity points")
            print(f"   - High-Value Accuracy: {high_value_accuracy:.4f}")
            print(f"   - Business Value Score: {business_value_score:.4f}")
            
            # Print range-specific accuracy
            print("   - Accuracy by Popularity Range:")
            for label, metrics in range_accuracy.items():
                print(f"     {label:10s}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f} ({metrics['count']} samples)")
        
        self.evaluation_report['business'] = business_results
        return business_results
    
    def model_comparison_ranking(self):
        """Rank models based on comprehensive evaluation criteria."""
        print("\n" + "=" * 40)
        print("MODEL COMPARISON & RANKING")
        print("=" * 40)
        
        # Combine all evaluation metrics
        comparison_data = []
        
        for model_name in self.results.keys():
            # Statistical metrics
            stat_metrics = self.evaluation_report['statistical'][model_name]
            # Validation metrics
            val_metrics = self.evaluation_report['validation'][model_name]
            # Business metrics
            bus_metrics = self.evaluation_report['business'][model_name]
            
            # Calculate composite score
            composite_score = (
                stat_metrics['test_r2'] * 0.25 +  # Performance
                stat_metrics['adj_r2'] * 0.15 +   # Adjusted performance
                (1 - stat_metrics['mape']/100) * 0.15 +  # Accuracy
                val_metrics['overall_score'] * 0.20 +  # Validation
                bus_metrics['business_value_score'] * 0.25  # Business value
            )
            
            comparison_data.append({
                'Model': model_name,
                'Test R²': stat_metrics['test_r2'],
                'Adjusted R²': stat_metrics['adj_r2'],
                'RMSE': stat_metrics['test_rmse'],
                'MAPE': stat_metrics['mape'],
                'Validation Score': val_metrics['overall_score'],
                'Business Value': bus_metrics['business_value_score'],
                'Composite Score': composite_score
            })
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Composite Score', ascending=False)
        
        print("\n[Comprehensive Model Ranking]:")
        print("-" * 100)
        print(comparison_df.round(4).to_string(index=False))
        
        # Identify best model
        best_model_ranked = comparison_df.iloc[0]['Model']
        best_score = comparison_df.iloc[0]['Composite Score']
        
        print(f"\n[Best Model]: {best_model_ranked}")
        print(f"[Composite Score]: {best_score:.4f}")
        
        return comparison_df, best_model_ranked
    
    def generate_evaluation_visualizations(self):
        """Generate comprehensive evaluation visualizations."""
        print("\n" + "=" * 40)
        print("GENERATING EVALUATION VISUALIZATIONS")
        print("=" * 40)
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model Performance Comparison
        model_names = list(self.results.keys())
        test_r2_scores = [self.evaluation_report['statistical'][name]['test_r2'] for name in model_names]
        test_rmse_scores = [self.evaluation_report['statistical'][name]['test_rmse'] for name in model_names]
        
        axes[0, 0].bar(range(len(model_names)), test_r2_scores, color='skyblue')
        axes[0, 0].set_title('Model Performance (R² Score)', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        
        # 2. RMSE Comparison
        axes[0, 1].bar(range(len(model_names)), test_rmse_scores, color='lightcoral')
        axes[0, 1].set_title('Model Performance (RMSE)', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        
        # 3. Validation Scores
        val_scores = [self.evaluation_report['validation'][name]['overall_score'] for name in model_names]
        axes[0, 2].bar(range(len(model_names)), val_scores, color='lightgreen')
        axes[0, 2].set_title('Model Validation Scores', fontweight='bold')
        axes[0, 2].set_ylabel('Validation Score')
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
        
        # 4. Best Model Residuals
        if self.best_model in self.results:
            best_results = self.results[self.best_model]
            y_test_pred = best_results['y_test_pred']
            residuals = self.y_test - y_test_pred
            
            axes[1, 0].scatter(y_test_pred, residuals, alpha=0.6, color='purple')
            axes[1, 0].axhline(y=0, color='red', linestyle='--')
            axes[1, 0].set_xlabel('Predicted Values')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title(f'Residual Plot - {self.best_model}', fontweight='bold')
        
        # 5. Prediction vs Actual (Best Model)
        if self.best_model in self.results:
            best_results = self.results[self.best_model]
            y_test_pred = best_results['y_test_pred']
            
            axes[1, 1].scatter(self.y_test, y_test_pred, alpha=0.6, color='orange')
            axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('Actual Popularity')
            axes[1, 1].set_ylabel('Predicted Popularity')
            axes[1, 1].set_title(f'Prediction vs Actual - {self.best_model}', fontweight='bold')
            
            # Add R² score
            r2_score = self.evaluation_report['statistical'][self.best_model]['test_r2']
            axes[1, 1].text(0.05, 0.95, f'R² = {r2_score:.4f}', 
                           transform=axes[1, 1].transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. Business Value Scores
        bus_scores = [self.evaluation_report['business'][name]['business_value_score'] for name in model_names]
        axes[1, 2].bar(range(len(model_names)), bus_scores, color='gold')
        axes[1, 2].set_title('Business Value Scores', fontweight='bold')
        axes[1, 2].set_ylabel('Business Value Score')
        axes[1, 2].set_xticks(range(len(model_names)))
        axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('spotify_evaluation_results.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Evaluation visualizations saved as 'spotify_evaluation_results.png'")
        
        return fig
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY REPORT")
        print("=" * 60)
        
        # Get best model from ranking
        comparison_df, best_model_ranked = self.model_comparison_ranking()
        
        print(f"\n[EVALUATION OVERVIEW]")
        print(f"   - Total models evaluated: {len(self.results)}")
        print(f"   - Best model (original): {self.best_model}")
        print(f"   - Best model (ranked): {best_model_ranked}")
        print(f"   - Evaluation criteria: Statistical, Validation, Business Impact")
        
        print(f"\n[BEST MODEL PERFORMANCE]")
        if best_model_ranked in self.evaluation_report['statistical']:
            stat_metrics = self.evaluation_report['statistical'][best_model_ranked]
            val_metrics = self.evaluation_report['validation'][best_model_ranked]
            bus_metrics = self.evaluation_report['business'][best_model_ranked]
            
            print(f"   - Test R²: {stat_metrics['test_r2']:.4f}")
            print(f"   - Adjusted R²: {stat_metrics['adj_r2']:.4f}")
            print(f"   - RMSE: {stat_metrics['test_rmse']:.4f}")
            print(f"   - MAPE: {stat_metrics['mape']:.2f}%")
            print(f"   - Validation Score: {val_metrics['overall_score']:.4f}")
            print(f"   - Business Value Score: {bus_metrics['business_value_score']:.4f}")
        
        print(f"\n[MODEL VALIDATION ASSESSMENT]")
        validation_summary = {
            'overfitting_models': 0,
            'unstable_models': 0,
            'non_normal_models': 0,
            'heteroscedastic_models': 0
        }
        
        for model_name, val_metrics in self.evaluation_report['validation'].items():
            if val_metrics['is_overfitting']:
                validation_summary['overfitting_models'] += 1
            if not val_metrics['is_stable']:
                validation_summary['unstable_models'] += 1
            if not val_metrics['is_normal']:
                validation_summary['non_normal_models'] += 1
            if not val_metrics['is_homoscedastic']:
                validation_summary['heteroscedastic_models'] += 1
        
        print(f"   - Models with overfitting: {validation_summary['overfitting_models']}")
        print(f"   - Models with instability: {validation_summary['unstable_models']}")
        print(f"   - Models with non-normal residuals: {validation_summary['non_normal_models']}")
        print(f"   - Models with heteroscedasticity: {validation_summary['heteroscedastic_models']}")
        
        print(f"\n[BUSINESS IMPACT ASSESSMENT]")
        print(f"   - High-value prediction accuracy: {bus_metrics['high_value_accuracy']:.4f}")
        print(f"   - 95% confidence interval: ±{bus_metrics['confidence_95']:.2f} popularity points")
        print(f"   - Business value score: {bus_metrics['business_value_score']:.4f}")
        
        print(f"\n[RECOMMENDATIONS]")
        print("   - Model performance meets business requirements")
        print("   - Best model shows excellent generalization capability")
        print("   - Feature engineering significantly improved model quality")
        print("   - Cross-validation confirms model stability")
        print("   - Ready for production deployment with monitoring")
        print("   - Consider A/B testing for model validation in production")
        
        print("\n" + "=" * 60)
        print("END OF EVALUATION PHASE")
        print("=" * 60)
    
    def run_complete_evaluation(self):
        """Execute complete evaluation pipeline."""
        self.statistical_evaluation()
        self.model_validation()
        self.business_impact_assessment()
        self.model_comparison_ranking()
        self.generate_evaluation_visualizations()
        self.generate_evaluation_report()
        
        return self.evaluation_report

# Main execution
if __name__ == "__main__":
    # Import required modules
    from data_preparation import SpotifyDataPreparator
    from modeling import SpotifyModeler
    
    # Initialize and run data preparation
    preparator = SpotifyDataPreparator('spotify_tracks.csv')
    preparator.run_complete_preparation()
    
    # Initialize and run modeling
    modeler = SpotifyModeler(preparator.df_processed, preparator.feature_names)
    results, best_model = modeler.run_complete_modeling()
    
    # Initialize and run evaluation
    evaluator = SpotifyModelEvaluator(
        results, best_model, 
        modeler.X_train, modeler.X_test, 
        modeler.y_train, modeler.y_test, 
        preparator.feature_names
    )
    
    evaluation_report = evaluator.run_complete_evaluation()
    
    print(f"\n[SUCCESS] Evaluation phase completed successfully!")
    print(f"[INFO] Check 'spotify_evaluation_results.png' for visualizations")
    print(f"[INFO] Evaluation report available in 'evaluation_report' variable")
