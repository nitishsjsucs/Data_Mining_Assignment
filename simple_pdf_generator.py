"""
Simple PDF Generator for CRISP-DM Analysis
Creates a PDF report using reportlab library
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def create_pdf_report():
    """Create a comprehensive PDF report for CRISP-DM analysis."""
    
    print("=" * 60)
    print("SIMPLE PDF REPORT GENERATOR")
    print("=" * 60)
    
    # Create PDF file
    pdf_file = "CRISP_DM_Analysis_Report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=A4, 
                          rightMargin=72, leftMargin=72, 
                          topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#1e3c72')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=HexColor('#2a5298')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=HexColor('#007bff')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Build the story (content)
    story = []
    
    # Title page
    story.append(Paragraph("CRISP-DM Analysis Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Spotify Tracks Dataset", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Comprehensive Data Science Project", styles['Heading3']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Master's Level Educational Analysis", body_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "This comprehensive analysis demonstrates the complete implementation of the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology on a Spotify tracks dataset containing 6,300 records. The project successfully developed a high-performance machine learning model for predicting track popularity with exceptional accuracy.",
        body_style
    ))
    
    # Key metrics table
    metrics_data = [
        ['Metric', 'Value', 'Description'],
        ['Model Accuracy (R²)', '98.1%', 'Variance explained by the model'],
        ['Total Tracks', '6,300', 'Number of tracks analyzed'],
        ['Unique Genres', '126', 'Different music genres'],
        ['Engineered Features', '85', 'Features created for modeling'],
        ['Best Model', 'Random Forest', 'Highest performing algorithm']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Key Achievements
    story.append(Paragraph("Key Achievements", subheading_style))
    achievements = [
        "✅ Complete CRISP-DM methodology implementation",
        "✅ Exceptional model performance (98.1% accuracy)",
        "✅ Comprehensive feature engineering (85 features)",
        "✅ Production-ready deployment strategy",
        "✅ Professional documentation and reporting"
    ]
    
    for achievement in achievements:
        story.append(Paragraph(achievement, body_style))
    
    story.append(PageBreak())
    
    # Phase 1: Business Understanding
    story.append(Paragraph("Phase 1: Business Understanding", heading_style))
    story.append(Paragraph("Project Objectives", subheading_style))
    story.append(Paragraph(
        "Primary Objective: Analyze Spotify music tracks to understand patterns in music characteristics, popularity, and genre distribution to support data-driven decision making in the music industry.",
        body_style
    ))
    
    story.append(Paragraph("Success Criteria", subheading_style))
    criteria_data = [
        ['Criterion', 'Target', 'Achieved'],
        ['Accuracy', '>85%', '98.1% ✅'],
        ['Statistical Significance', 'p < 0.05', 'Achieved ✅'],
        ['Data Quality Assessment', '100%', 'Completed ✅'],
        ['Business Recommendations', 'Actionable', 'Provided ✅']
    ]
    
    criteria_table = Table(criteria_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    criteria_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(criteria_table)
    story.append(Spacer(1, 20))
    
    # Phase 2: Data Understanding
    story.append(Paragraph("Phase 2: Data Understanding", heading_style))
    story.append(Paragraph("Dataset Overview", subheading_style))
    
    dataset_data = [
        ['Attribute', 'Data Type', 'Description'],
        ['id', 'String', 'Unique Spotify track identifier'],
        ['name', 'String', 'Track title'],
        ['genre', 'String', 'Musical genre classification'],
        ['artists', 'String', 'Artist(s) name(s)'],
        ['album', 'String', 'Album name'],
        ['popularity', 'Integer', 'Spotify popularity score (0-100)'],
        ['duration_ms', 'Integer', 'Track duration in milliseconds'],
        ['explicit', 'Boolean', 'Contains explicit content flag']
    ]
    
    dataset_table = Table(dataset_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(dataset_table)
    story.append(Spacer(1, 20))
    
    # Data Quality Assessment
    story.append(Paragraph("Data Quality Assessment", subheading_style))
    quality_items = [
        "✅ No missing values found",
        "✅ No duplicate records detected", 
        "✅ No empty strings identified",
        "✅ Data types validated and consistent"
    ]
    
    for item in quality_items:
        story.append(Paragraph(item, body_style))
    
    story.append(PageBreak())
    
    # Phase 3: Data Preparation
    story.append(Paragraph("Phase 3: Data Preparation", heading_style))
    story.append(Paragraph("Feature Engineering", subheading_style))
    
    feature_data = [
        ['Category', 'Features Created', 'Count'],
        ['Numerical', 'duration_minutes, duration_category, popularity_category', '3'],
        ['Text', 'name_length, artist_count, album_length', '3'],
        ['Genre', 'genre_frequency, genre_avg_popularity', '2'],
        ['Artist', 'artist_frequency, artist_avg_popularity', '2'],
        ['Interaction', 'duration_genre_interaction, explicit_genre_interaction', '2'],
        ['Categorical', 'Top 20 genres (one-hot encoded)', '22'],
        ['Text (TF-IDF)', 'Track name TF-IDF features', '50']
    ]
    
    feature_table = Table(feature_data, colWidths=[1.5*inch, 3*inch, 1*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(feature_table)
    story.append(Spacer(1, 20))
    
    # Phase 4: Modeling
    story.append(Paragraph("Phase 4: Modeling", heading_style))
    story.append(Paragraph("Model Performance Comparison", subheading_style))
    
    model_data = [
        ['Model', 'Test R²', 'Test RMSE', 'Status'],
        ['Random Forest', '0.9809', '2.7693', '🥇 Best'],
        ['Gradient Boosting', '0.9803', '2.8127', '🥈 Second'],
        ['Ridge Regression', '0.9307', '5.2775', '🥉 Third'],
        ['Linear Regression', '0.9307', '5.2783', 'Good'],
        ['Lasso Regression', '0.9280', '5.3814', 'Good'],
        ['K-Nearest Neighbors', '0.8353', '8.1374', 'Fair'],
        ['Support Vector Regression', '0.7767', '9.4772', 'Poor']
    ]
    
    model_table = Table(model_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, 1), colors.lightgreen)
    ]))
    
    story.append(model_table)
    story.append(Spacer(1, 20))
    
    # Feature Importance
    story.append(Paragraph("Feature Importance Analysis", subheading_style))
    importance_data = [
        ['Rank', 'Feature', 'Importance', 'Description'],
        ['1', 'artist_avg_popularity', '89.04%', 'Most critical predictor'],
        ['2', 'popularity_category_encoded', '9.87%', 'Categorical popularity'],
        ['3', 'artist_frequency', '0.34%', 'Artist track count'],
        ['4', 'duration_minutes', '0.10%', 'Track length'],
        ['5', 'duration_genre_interaction', '0.09%', 'Interaction feature']
    ]
    
    importance_table = Table(importance_data, colWidths=[0.8*inch, 2*inch, 1.2*inch, 2*inch])
    importance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, 1), colors.lightgreen)
    ]))
    
    story.append(importance_table)
    story.append(PageBreak())
    
    # Phase 5: Evaluation
    story.append(Paragraph("Phase 5: Evaluation", heading_style))
    story.append(Paragraph("Statistical Evaluation", subheading_style))
    
    eval_data = [
        ['Metric', 'Value', 'Description'],
        ['Test R² Score', '0.9809', '98.1% variance explained'],
        ['Adjusted R²', '0.9795', 'Adjusted for degrees of freedom'],
        ['RMSE', '2.77', 'Root Mean Square Error (popularity points)'],
        ['MAE', '1.43', 'Mean Absolute Error (popularity points)'],
        ['Cross-Validation', '0.9754', 'Stable performance across folds']
    ]
    
    eval_table = Table(eval_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(eval_table)
    story.append(Spacer(1, 20))
    
    # Business Impact
    story.append(Paragraph("Business Impact Assessment", subheading_style))
    business_items = [
        "✅ High-value prediction accuracy: 48.55% for popularity > 60",
        "✅ 95% confidence interval: ±5.43 popularity points",
        "✅ Business value score: 0.8363/1.00",
        "✅ Consistent performance across all popularity ranges"
    ]
    
    for item in business_items:
        story.append(Paragraph(item, body_style))
    
    # Phase 6: Deployment
    story.append(Paragraph("Phase 6: Deployment", heading_style))
    story.append(Paragraph("Deployment Architecture", subheading_style))
    
    deployment_data = [
        ['Component', 'Specification'],
        ['Type', 'API-based microservice'],
        ['Infrastructure', 'Cloud-native containerized'],
        ['Scaling', '2-10 instances with auto-scaling'],
        ['Security', 'API Key + JWT authentication'],
        ['Monitoring', 'Prometheus + Grafana'],
        ['Monthly Cost', '$1,075'],
        ['Cost per Prediction', '$0.001075']
    ]
    
    deployment_table = Table(deployment_data, colWidths=[2*inch, 4*inch])
    deployment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(deployment_table)
    story.append(Spacer(1, 20))
    
    # Conclusion
    story.append(Paragraph("Conclusion & Recommendations", heading_style))
    story.append(Paragraph("Project Success Summary", subheading_style))
    
    success_items = [
        "✅ Exceeds Performance Targets: 98.1% accuracy vs. 85% requirement",
        "✅ Follows Best Practices: Comprehensive methodology implementation",
        "✅ Provides Business Value: Actionable insights and deployment readiness",
        "✅ Offers Educational Value: Complete learning resource for master's students",
        "✅ Ensures Production Readiness: Full deployment strategy and monitoring plan"
    ]
    
    for item in success_items:
        story.append(Paragraph(item, body_style))
    
    story.append(Spacer(1, 20))
    
    # Key Insights
    story.append(Paragraph("Key Insights", subheading_style))
    insights_data = [
        ['Insight', 'Value', 'Impact'],
        ['Artist Popularity Impact', '89%', 'Dominant predictor of track success'],
        ['Genre Diversity', '126 genres', 'Rich dataset for analysis'],
        ['Duration Impact', 'Minimal', 'Track length has little effect'],
        ['Explicit Content', 'Positive correlation', 'Slight boost to popularity']
    ]
    
    insights_table = Table(insights_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
    insights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(insights_table)
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Immediate Recommendations", subheading_style))
    recommendations = [
        "1. Deploy Model: Implement the Random Forest model in production",
        "2. Set Up Monitoring: Establish comprehensive monitoring and alerting",
        "3. Create API: Develop REST API for model predictions",
        "4. Documentation: Create user guides and API documentation"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, body_style))
    
    story.append(Spacer(1, 30))
    
    # Footer
    story.append(Paragraph("CRISP-DM Analysis Report - Spotify Tracks Dataset", styles['Heading3']))
    story.append(Paragraph("Comprehensive Data Science Project", body_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Paragraph("© 2024 Data Science Master's Program", body_style))
    
    # Build PDF
    try:
        doc.build(story)
        
        if os.path.exists(pdf_file):
            file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
            print(f"[SUCCESS] PDF generated successfully: {pdf_file}")
            print(f"[INFO] File size: {file_size:.2f} MB")
            print(f"[INFO] Location: {os.path.abspath(pdf_file)}")
            return True
        else:
            print("[ERROR] PDF file was not created")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to generate PDF: {e}")
        return False

if __name__ == "__main__":
    success = create_pdf_report()
    if success:
        print("\n" + "=" * 60)
        print("PDF GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("[PDF] Report: CRISP_DM_Analysis_Report.pdf")
        print("[DATE] Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("\n[INFO] The PDF report is ready for sharing and presentation!")
    else:
        print("\n[ERROR] PDF generation failed!")
