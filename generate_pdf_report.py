"""
PDF Report Generator for CRISP-DM Analysis
Converts the HTML report to a professional PDF document
"""

import os
import sys
from datetime import datetime

def generate_pdf_report():
    """Generate PDF report from HTML using weasyprint or alternative methods."""
    
    print("=" * 60)
    print("PDF REPORT GENERATOR")
    print("=" * 60)
    
    # Check if HTML file exists
    html_file = "CRISP_DM_Analysis_Report.html"
    if not os.path.exists(html_file):
        print(f"[ERROR] HTML file '{html_file}' not found!")
        return False
    
    print(f"[INFO] Found HTML file: {html_file}")
    
    # Try different PDF generation methods
    pdf_generated = False
    
    # Method 1: Try weasyprint
    try:
        import weasyprint
        print("[INFO] Using WeasyPrint for PDF generation...")
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Generate PDF
        pdf_file = "CRISP_DM_Analysis_Report.pdf"
        weasyprint.HTML(string=html_content).write_pdf(pdf_file)
        
        if os.path.exists(pdf_file):
            file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
            print(f"[SUCCESS] PDF generated successfully: {pdf_file}")
            print(f"[INFO] File size: {file_size:.2f} MB")
            pdf_generated = True
        else:
            print("[ERROR] PDF file was not created")
            
    except ImportError:
        print("[WARNING] WeasyPrint not available, trying alternative method...")
        
        # Method 2: Try pdfkit (wkhtmltopdf)
        try:
            import pdfkit
            print("[INFO] Using pdfkit for PDF generation...")
            
            # Configure options for better PDF output
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None,
                'enable-local-file-access': None,
                'print-media-type': None,
                'disable-smart-shrinking': None,
                'zoom': '0.8'
            }
            
            pdf_file = "CRISP_DM_Analysis_Report.pdf"
            pdfkit.from_file(html_file, pdf_file, options=options)
            
            if os.path.exists(pdf_file):
                file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
                print(f"[SUCCESS] PDF generated successfully: {pdf_file}")
                print(f"[INFO] File size: {file_size:.2f} MB")
                pdf_generated = True
            else:
                print("[ERROR] PDF file was not created")
                
        except ImportError:
            print("[WARNING] pdfkit not available, trying alternative method...")
            
            # Method 3: Try playwright
            try:
                from playwright.sync_api import sync_playwright
                print("[INFO] Using Playwright for PDF generation...")
                
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    page = browser.new_page()
                    
                    # Load HTML file
                    page.goto(f"file://{os.path.abspath(html_file)}")
                    
                    # Wait for content to load
                    page.wait_for_load_state('networkidle')
                    
                    # Generate PDF
                    pdf_file = "CRISP_DM_Analysis_Report.pdf"
                    page.pdf(
                        path=pdf_file,
                        format='A4',
                        margin={
                            'top': '0.75in',
                            'right': '0.75in',
                            'bottom': '0.75in',
                            'left': '0.75in'
                        },
                        print_background=True
                    )
                    
                    browser.close()
                    
                    if os.path.exists(pdf_file):
                        file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
                        print(f"[SUCCESS] PDF generated successfully: {pdf_file}")
                        print(f"[INFO] File size: {file_size:.2f} MB")
                        pdf_generated = True
                    else:
                        print("[ERROR] PDF file was not created")
                        
            except ImportError:
                print("[WARNING] Playwright not available, trying alternative method...")
                
                # Method 4: Try selenium
                try:
                    from selenium import webdriver
                    from selenium.webdriver.chrome.options import Options
                    print("[INFO] Using Selenium for PDF generation...")
                    
                    # Configure Chrome options
                    chrome_options = Options()
                    chrome_options.add_argument('--headless')
                    chrome_options.add_argument('--no-sandbox')
                    chrome_options.add_argument('--disable-dev-shm-usage')
                    chrome_options.add_argument('--disable-gpu')
                    chrome_options.add_argument('--window-size=1920,1080')
                    
                    # Create driver
                    driver = webdriver.Chrome(options=chrome_options)
                    
                    # Load HTML file
                    driver.get(f"file://{os.path.abspath(html_file)}")
                    
                    # Wait for content to load
                    import time
                    time.sleep(3)
                    
                    # Generate PDF using Chrome's print functionality
                    pdf_file = "CRISP_DM_Analysis_Report.pdf"
                    
                    # Execute JavaScript to print to PDF
                    print_script = """
                    window.print();
                    """
                    driver.execute_script(print_script)
                    
                    # Note: This method requires manual intervention
                    print("[INFO] Please save the PDF manually from the print dialog")
                    print("[INFO] Press Enter when done...")
                    input()
                    
                    driver.quit()
                    pdf_generated = True
                    
                except ImportError:
                    print("[ERROR] No PDF generation libraries available")
                    print("[INFO] Please install one of the following:")
                    print("  - weasyprint: pip install weasyprint")
                    print("  - pdfkit: pip install pdfkit (requires wkhtmltopdf)")
                    print("  - playwright: pip install playwright")
                    print("  - selenium: pip install selenium")
                    return False
    
    if pdf_generated:
        print("\n" + "=" * 60)
        print("PDF GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"📄 PDF Report: CRISP_DM_Analysis_Report.pdf")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Location: {os.path.abspath('CRISP_DM_Analysis_Report.pdf')}")
        print("\n[INFO] The PDF report is ready for sharing and presentation!")
        return True
    else:
        print("\n[ERROR] PDF generation failed!")
        return False

def install_dependencies():
    """Install required dependencies for PDF generation."""
    print("=" * 60)
    print("INSTALLING PDF GENERATION DEPENDENCIES")
    print("=" * 60)
    
    dependencies = [
        "weasyprint",
        "pdfkit",
        "playwright",
        "selenium"
    ]
    
    for dep in dependencies:
        try:
            print(f"[INFO] Installing {dep}...")
            os.system(f"pip install {dep}")
            print(f"[SUCCESS] {dep} installed successfully")
        except Exception as e:
            print(f"[WARNING] Failed to install {dep}: {e}")
    
    print("\n[INFO] Dependencies installation completed!")
    print("[INFO] You can now run the PDF generation script.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        install_dependencies()
    else:
        success = generate_pdf_report()
        if not success:
            print("\n[INFO] To install dependencies, run:")
            print("python generate_pdf_report.py install")
