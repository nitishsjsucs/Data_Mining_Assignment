"""
Comprehensive Deployment Strategy for Spotify Tracks Dataset
CRISP-DM Phase 6: Deployment

This script provides a complete deployment strategy, monitoring plan, and implementation
guidelines following CRISP-DM methodology and professional data science practices.
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SpotifyModelDeployment:
    """
    Comprehensive deployment strategy class for Spotify tracks dataset.
    Implements professional deployment practices following CRISP-DM methodology.
    """
    
    def __init__(self, best_model, feature_names, preprocessing_pipeline=None):
        """Initialize the deployment strategy with best model and features."""
        self.best_model = best_model
        self.feature_names = feature_names
        self.preprocessing_pipeline = preprocessing_pipeline
        self.deployment_config = {}
        self.monitoring_plan = {}
        
    def model_packaging(self):
        """Package model and preprocessing pipeline for deployment."""
        print("=" * 60)
        print("CRISP-DM PHASE 6: DEPLOYMENT")
        print("=" * 60)
        
        print("\n" + "=" * 40)
        print("MODEL PACKAGING")
        print("=" * 40)
        
        # Create model package
        model_package = {
            'model': self.best_model,
            'feature_names': self.feature_names,
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'model_metadata': {
                'model_type': type(self.best_model).__name__,
                'feature_count': len(self.feature_names),
                'creation_date': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        # Save model package
        with open('spotify_popularity_model.pkl', 'wb') as f:
            pickle.dump(model_package, f)
        
        print("[SUCCESS] Model packaged and saved as 'spotify_popularity_model.pkl'")
        print(f"   - Model type: {type(self.best_model).__name__}")
        print(f"   - Features: {len(self.feature_names)}")
        print(f"   - Package size: {self._get_file_size('spotify_popularity_model.pkl')} MB")
        
        return model_package
    
    def deployment_architecture(self):
        """Design deployment architecture and infrastructure."""
        print("\n" + "=" * 40)
        print("DEPLOYMENT ARCHITECTURE")
        print("=" * 40)
        
        architecture = {
            'deployment_type': 'API-based microservice',
            'infrastructure': {
                'cloud_provider': 'AWS/Azure/GCP',
                'compute': 'Containerized (Docker)',
                'orchestration': 'Kubernetes',
                'load_balancer': 'Application Load Balancer',
                'database': 'PostgreSQL for metadata',
                'cache': 'Redis for model caching',
                'monitoring': 'Prometheus + Grafana'
            },
            'scalability': {
                'horizontal_scaling': True,
                'auto_scaling': True,
                'min_instances': 2,
                'max_instances': 10,
                'cpu_threshold': 70,
                'memory_threshold': 80
            },
            'security': {
                'authentication': 'API Key + JWT',
                'encryption': 'TLS 1.3',
                'rate_limiting': '100 requests/minute per API key',
                'input_validation': 'Strict schema validation',
                'logging': 'Structured logging with PII masking'
            }
        }
        
        print("[SUCCESS] Deployment architecture designed:")
        print(f"   - Type: {architecture['deployment_type']}")
        print(f"   - Infrastructure: {architecture['infrastructure']['cloud_provider']}")
        print(f"   - Scaling: {architecture['scalability']['min_instances']}-{architecture['scalability']['max_instances']} instances")
        print(f"   - Security: {architecture['security']['authentication']}")
        
        self.deployment_config['architecture'] = architecture
        return architecture
    
    def api_design(self):
        """Design REST API for model deployment."""
        print("\n" + "=" * 40)
        print("API DESIGN")
        print("=" * 40)
        
        api_spec = {
            'base_url': 'https://api.spotify-predictor.com/v1',
            'endpoints': {
                'predict': {
                    'path': '/predict',
                    'method': 'POST',
                    'description': 'Predict track popularity',
                    'request_schema': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string', 'description': 'Track name'},
                            'genre': {'type': 'string', 'description': 'Music genre'},
                            'artists': {'type': 'string', 'description': 'Artist names'},
                            'album': {'type': 'string', 'description': 'Album name'},
                            'duration_ms': {'type': 'integer', 'description': 'Duration in milliseconds'},
                            'explicit': {'type': 'boolean', 'description': 'Explicit content flag'}
                        },
                        'required': ['name', 'genre', 'artists', 'album', 'duration_ms', 'explicit']
                    },
                    'response_schema': {
                        'type': 'object',
                        'properties': {
                            'prediction': {'type': 'number', 'description': 'Predicted popularity (0-100)'},
                            'confidence': {'type': 'number', 'description': 'Prediction confidence (0-1)'},
                            'model_version': {'type': 'string', 'description': 'Model version used'},
                            'processing_time_ms': {'type': 'number', 'description': 'Processing time in milliseconds'}
                        }
                    }
                },
                'health': {
                    'path': '/health',
                    'method': 'GET',
                    'description': 'Health check endpoint',
                    'response_schema': {
                        'type': 'object',
                        'properties': {
                            'status': {'type': 'string', 'enum': ['healthy', 'unhealthy']},
                            'timestamp': {'type': 'string', 'format': 'date-time'},
                            'model_version': {'type': 'string'},
                            'uptime_seconds': {'type': 'number'}
                        }
                    }
                },
                'metrics': {
                    'path': '/metrics',
                    'method': 'GET',
                    'description': 'Model performance metrics',
                    'response_schema': {
                        'type': 'object',
                        'properties': {
                            'total_predictions': {'type': 'integer'},
                            'average_processing_time_ms': {'type': 'number'},
                            'error_rate': {'type': 'number'},
                            'model_accuracy': {'type': 'number'}
                        }
                    }
                }
            },
            'rate_limiting': {
                'requests_per_minute': 100,
                'burst_limit': 20,
                'per_api_key': True
            },
            'error_handling': {
                '400': 'Bad Request - Invalid input data',
                '401': 'Unauthorized - Invalid API key',
                '429': 'Too Many Requests - Rate limit exceeded',
                '500': 'Internal Server Error - Model prediction failed',
                '503': 'Service Unavailable - Model not ready'
            }
        }
        
        print("[SUCCESS] API specification designed:")
        print(f"   - Base URL: {api_spec['base_url']}")
        print(f"   - Endpoints: {len(api_spec['endpoints'])}")
        print(f"   - Rate limiting: {api_spec['rate_limiting']['requests_per_minute']} req/min")
        
        self.deployment_config['api'] = api_spec
        return api_spec
    
    def monitoring_strategy(self):
        """Design comprehensive monitoring and alerting strategy."""
        print("\n" + "=" * 40)
        print("MONITORING STRATEGY")
        print("=" * 40)
        
        monitoring_plan = {
            'metrics': {
                'performance': {
                    'prediction_latency': {
                        'threshold': 100,  # milliseconds
                        'alert_level': 'warning'
                    },
                    'throughput': {
                        'threshold': 50,  # requests per second
                        'alert_level': 'info'
                    },
                    'error_rate': {
                        'threshold': 5,  # percentage
                        'alert_level': 'critical'
                    }
                },
                'model': {
                    'prediction_drift': {
                        'threshold': 0.1,  # statistical distance
                        'alert_level': 'warning'
                    },
                    'feature_drift': {
                        'threshold': 0.05,  # statistical distance
                        'alert_level': 'warning'
                    },
                    'model_accuracy': {
                        'threshold': 0.95,  # minimum R² score
                        'alert_level': 'critical'
                    }
                },
                'infrastructure': {
                    'cpu_usage': {
                        'threshold': 80,  # percentage
                        'alert_level': 'warning'
                    },
                    'memory_usage': {
                        'threshold': 85,  # percentage
                        'alert_level': 'warning'
                    },
                    'disk_usage': {
                        'threshold': 90,  # percentage
                        'alert_level': 'critical'
                    }
                }
            },
            'alerting': {
                'channels': ['email', 'slack', 'pagerduty'],
                'escalation': {
                    'level_1': 'Data Science Team',
                    'level_2': 'Engineering Team',
                    'level_3': 'On-call Engineer'
                },
                'schedules': {
                    'business_hours': '9 AM - 6 PM EST',
                    'after_hours': '6 PM - 9 AM EST',
                    'weekends': '24/7 coverage'
                }
            },
            'dashboards': {
                'executive': {
                    'metrics': ['total_predictions', 'average_accuracy', 'uptime'],
                    'refresh_rate': '5 minutes'
                },
                'operational': {
                    'metrics': ['latency', 'error_rate', 'throughput', 'resource_usage'],
                    'refresh_rate': '1 minute'
                },
                'model': {
                    'metrics': ['prediction_drift', 'feature_drift', 'model_accuracy'],
                    'refresh_rate': '1 hour'
                }
            }
        }
        
        print("[SUCCESS] Monitoring strategy designed:")
        print(f"   - Metrics categories: {len(monitoring_plan['metrics'])}")
        print(f"   - Alert channels: {len(monitoring_plan['alerting']['channels'])}")
        print(f"   - Dashboard types: {len(monitoring_plan['dashboards'])}")
        
        self.monitoring_plan = monitoring_plan
        return monitoring_plan
    
    def deployment_phases(self):
        """Define phased deployment strategy."""
        print("\n" + "=" * 40)
        print("DEPLOYMENT PHASES")
        print("=" * 40)
        
        phases = {
            'phase_1': {
                'name': 'Development Environment',
                'duration': '1 week',
                'objectives': [
                    'Set up development infrastructure',
                    'Deploy model in development',
                    'Test API endpoints',
                    'Validate model performance'
                ],
                'success_criteria': [
                    'Model loads successfully',
                    'API responds within 100ms',
                    'All tests pass',
                    'Monitoring setup complete'
                ]
            },
            'phase_2': {
                'name': 'Staging Environment',
                'duration': '1 week',
                'objectives': [
                    'Deploy to staging environment',
                    'Load testing',
                    'Security testing',
                    'Integration testing'
                ],
                'success_criteria': [
                    'Handles 1000 requests/minute',
                    'Security scan passes',
                    'Integration tests pass',
                    'Performance meets SLA'
                ]
            },
            'phase_3': {
                'name': 'Production Deployment',
                'duration': '2 weeks',
                'objectives': [
                    'Deploy to production',
                    'A/B testing setup',
                    'Gradual traffic increase',
                    'Monitor performance'
                ],
                'success_criteria': [
                    'Zero downtime deployment',
                    'A/B test shows no regression',
                    'Traffic increased to 100%',
                    'All metrics within thresholds'
                ]
            },
            'phase_4': {
                'name': 'Post-Deployment',
                'duration': 'Ongoing',
                'objectives': [
                    'Continuous monitoring',
                    'Performance optimization',
                    'Model retraining',
                    'Feature updates'
                ],
                'success_criteria': [
                    'Model accuracy maintained',
                    'No critical alerts',
                    'Regular retraining schedule',
                    'Feature updates deployed'
                ]
            }
        }
        
        print("[SUCCESS] Deployment phases defined:")
        for phase, details in phases.items():
            print(f"   - {details['name']}: {details['duration']}")
        
        self.deployment_config['phases'] = phases
        return phases
    
    def risk_assessment(self):
        """Assess deployment risks and mitigation strategies."""
        print("\n" + "=" * 40)
        print("RISK ASSESSMENT")
        print("=" * 40)
        
        risks = {
            'high': {
                'model_degradation': {
                    'description': 'Model performance degrades over time',
                    'probability': 'Medium',
                    'impact': 'High',
                    'mitigation': [
                        'Implement model monitoring',
                        'Set up automated retraining',
                        'Define performance thresholds',
                        'Create rollback procedures'
                    ]
                },
                'data_drift': {
                    'description': 'Input data distribution changes',
                    'probability': 'High',
                    'impact': 'Medium',
                    'mitigation': [
                        'Monitor feature distributions',
                        'Implement drift detection',
                        'Update preprocessing pipeline',
                        'Retrain model when needed'
                    ]
                }
            },
            'medium': {
                'scalability_issues': {
                    'description': 'System cannot handle increased load',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'mitigation': [
                        'Implement auto-scaling',
                        'Load testing',
                        'Performance monitoring',
                        'Capacity planning'
                    ]
                },
                'security_vulnerabilities': {
                    'description': 'Security vulnerabilities in API',
                    'probability': 'Low',
                    'impact': 'High',
                    'mitigation': [
                        'Regular security audits',
                        'Input validation',
                        'Rate limiting',
                        'Authentication and authorization'
                    ]
                }
            },
            'low': {
                'infrastructure_failure': {
                    'description': 'Cloud infrastructure failures',
                    'probability': 'Low',
                    'impact': 'Medium',
                    'mitigation': [
                        'Multi-region deployment',
                        'Backup systems',
                        'Disaster recovery plan',
                        'Monitoring and alerting'
                    ]
                }
            }
        }
        
        print("[SUCCESS] Risk assessment completed:")
        for risk_level, risk_list in risks.items():
            print(f"   - {risk_level.title()} risks: {len(risk_list)}")
        
        self.deployment_config['risks'] = risks
        return risks
    
    def cost_analysis(self):
        """Analyze deployment costs and optimization opportunities."""
        print("\n" + "=" * 40)
        print("COST ANALYSIS")
        print("=" * 40)
        
        cost_breakdown = {
            'infrastructure': {
                'compute': {
                    'description': 'Container instances',
                    'monthly_cost': 500,  # USD
                    'optimization': 'Auto-scaling, spot instances'
                },
                'storage': {
                    'description': 'Model storage, logs, metrics',
                    'monthly_cost': 50,  # USD
                    'optimization': 'Data lifecycle policies'
                },
                'network': {
                    'description': 'Load balancer, data transfer',
                    'monthly_cost': 100,  # USD
                    'optimization': 'CDN, compression'
                }
            },
            'monitoring': {
                'metrics': {
                    'description': 'Prometheus, Grafana',
                    'monthly_cost': 200,  # USD
                    'optimization': 'Metric retention policies'
                },
                'logging': {
                    'description': 'Centralized logging',
                    'monthly_cost': 150,  # USD
                    'optimization': 'Log aggregation, filtering'
                }
            },
            'security': {
                'authentication': {
                    'description': 'API keys, JWT tokens',
                    'monthly_cost': 50,  # USD
                    'optimization': 'Token caching'
                },
                'encryption': {
                    'description': 'TLS certificates, encryption',
                    'monthly_cost': 25,  # USD
                    'optimization': 'Certificate automation'
                }
            },
            'total_monthly': 1075,  # USD
            'optimization_potential': {
                'estimated_savings': 200,  # USD
                'optimization_areas': [
                    'Auto-scaling optimization',
                    'Spot instance usage',
                    'Data retention policies',
                    'Resource right-sizing'
                ]
            }
        }
        
        print("[SUCCESS] Cost analysis completed:")
        print(f"   - Total monthly cost: ${cost_breakdown['total_monthly']}")
        print(f"   - Optimization potential: ${cost_breakdown['optimization_potential']['estimated_savings']}")
        print(f"   - Cost per prediction: ${cost_breakdown['total_monthly'] / 1000000:.6f} (1M predictions)")
        
        self.deployment_config['costs'] = cost_breakdown
        return cost_breakdown
    
    def generate_deployment_plan(self):
        """Generate comprehensive deployment plan document."""
        print("\n" + "=" * 40)
        print("DEPLOYMENT PLAN GENERATION")
        print("=" * 40)
        
        deployment_plan = {
            'executive_summary': {
                'project': 'Spotify Track Popularity Prediction',
                'model_type': 'Random Forest Regressor',
                'deployment_type': 'API-based microservice',
                'estimated_cost': '$1,075/month',
                'timeline': '4 weeks',
                'success_metrics': [
                    'Model accuracy > 95%',
                    'API latency < 100ms',
                    'Uptime > 99.9%',
                    'Zero security incidents'
                ]
            },
            'technical_specifications': self.deployment_config,
            'monitoring_plan': self.monitoring_plan,
            'implementation_timeline': {
                'week_1': 'Development environment setup',
                'week_2': 'Staging deployment and testing',
                'week_3': 'Production deployment',
                'week_4': 'Monitoring and optimization'
            },
            'success_criteria': {
                'performance': {
                    'latency': '< 100ms',
                    'throughput': '> 1000 requests/minute',
                    'accuracy': '> 95%',
                    'uptime': '> 99.9%'
                },
                'business': {
                    'user_satisfaction': '> 4.5/5',
                    'adoption_rate': '> 80%',
                    'cost_efficiency': '< $0.001 per prediction'
                }
            },
            'rollback_plan': {
                'triggers': [
                    'Model accuracy drops below 90%',
                    'API latency exceeds 200ms',
                    'Error rate exceeds 5%',
                    'Security incident detected'
                ],
                'procedures': [
                    'Immediate traffic reduction',
                    'Model version rollback',
                    'Infrastructure scaling',
                    'Incident response activation'
                ]
            }
        }
        
        # Save deployment plan
        with open('spotify_deployment_plan.json', 'w') as f:
            json.dump(deployment_plan, f, indent=2, default=str)
        
        print("[SUCCESS] Deployment plan generated and saved as 'spotify_deployment_plan.json'")
        
        return deployment_plan
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment summary report."""
        print("\n" + "=" * 60)
        print("DEPLOYMENT SUMMARY REPORT")
        print("=" * 60)
        
        print(f"\n[DEPLOYMENT OVERVIEW]")
        print(f"   - Model type: {type(self.best_model).__name__}")
        print(f"   - Features: {len(self.feature_names)}")
        print(f"   - Deployment type: API-based microservice")
        print(f"   - Infrastructure: Cloud-native containerized")
        
        print(f"\n[ARCHITECTURE SUMMARY]")
        if 'architecture' in self.deployment_config:
            arch = self.deployment_config['architecture']
            print(f"   - Cloud provider: {arch['infrastructure']['cloud_provider']}")
            print(f"   - Scaling: {arch['scalability']['min_instances']}-{arch['scalability']['max_instances']} instances")
            print(f"   - Security: {arch['security']['authentication']}")
        
        print(f"\n[API SPECIFICATION]")
        if 'api' in self.deployment_config:
            api = self.deployment_config['api']
            print(f"   - Base URL: {api['base_url']}")
            print(f"   - Endpoints: {len(api['endpoints'])}")
            print(f"   - Rate limiting: {api['rate_limiting']['requests_per_minute']} req/min")
        
        print(f"\n[MONITORING STRATEGY]")
        if self.monitoring_plan:
            print(f"   - Metrics categories: {len(self.monitoring_plan['metrics'])}")
            print(f"   - Alert channels: {len(self.monitoring_plan['alerting']['channels'])}")
            print(f"   - Dashboard types: {len(self.monitoring_plan['dashboards'])}")
        
        print(f"\n[DEPLOYMENT PHASES]")
        if 'phases' in self.deployment_config:
            for phase, details in self.deployment_config['phases'].items():
                print(f"   - {details['name']}: {details['duration']}")
        
        print(f"\n[RISK ASSESSMENT]")
        if 'risks' in self.deployment_config:
            for risk_level, risk_list in self.deployment_config['risks'].items():
                print(f"   - {risk_level.title()} risks: {len(risk_list)}")
        
        print(f"\n[COST ANALYSIS]")
        if 'costs' in self.deployment_config:
            costs = self.deployment_config['costs']
            print(f"   - Total monthly cost: ${costs['total_monthly']}")
            print(f"   - Optimization potential: ${costs['optimization_potential']['estimated_savings']}")
        
        print(f"\n[SUCCESS CRITERIA]")
        print("   - Model accuracy > 95%")
        print("   - API latency < 100ms")
        print("   - Uptime > 99.9%")
        print("   - Zero security incidents")
        print("   - Cost efficiency < $0.001 per prediction")
        
        print(f"\n[RECOMMENDATIONS]")
        print("   - Implement phased deployment approach")
        print("   - Set up comprehensive monitoring from day one")
        print("   - Establish automated retraining pipeline")
        print("   - Create incident response procedures")
        print("   - Plan for regular model updates")
        print("   - Monitor business impact and user satisfaction")
        
        print("\n" + "=" * 60)
        print("END OF DEPLOYMENT PHASE")
        print("=" * 60)
    
    def _get_file_size(self, filename):
        """Get file size in MB."""
        try:
            import os
            size_bytes = os.path.getsize(filename)
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return 0.0
    
    def run_complete_deployment(self):
        """Execute complete deployment strategy."""
        self.model_packaging()
        self.deployment_architecture()
        self.api_design()
        self.monitoring_strategy()
        self.deployment_phases()
        self.risk_assessment()
        self.cost_analysis()
        self.generate_deployment_plan()
        self.generate_deployment_report()
        
        return self.deployment_config, self.monitoring_plan

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
    
    # Initialize and run deployment
    deployment = SpotifyModelDeployment(
        results[best_model]['model'], 
        preparator.feature_names
    )
    
    deployment_config, monitoring_plan = deployment.run_complete_deployment()
    
    print(f"\n[SUCCESS] Deployment phase completed successfully!")
    print(f"[INFO] Model packaged as 'spotify_popularity_model.pkl'")
    print(f"[INFO] Deployment plan saved as 'spotify_deployment_plan.json'")
    print(f"[INFO] Ready for production deployment")
