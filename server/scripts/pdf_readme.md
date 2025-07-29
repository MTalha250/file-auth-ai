## PDF Text Extraction Pipeline

A Python pipeline that extracts text from PDFs using a hybrid approach combining native text extraction with OCR for maximum text recovery.
  


## Quick Setup
1. Install Tesseract OCR
2. Install dependencies (provided at the end)
3. Run!


  
## Usage

from main import PDFTextExtractor

- Initialize
  
extractor = PDFTextExtractor(
    tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

- Process PDF
  
results = extractor.process_pdf("document.pdf")
output_file = extractor.save_extracted_text(results)


(also refer to provided sample main to know more)
  


## Processing Pipeline

1. Diagnosis: Analyzes PDF structure - counts pages, images, drawings, and annotations. Identifies image properties and text-rich vs image-heavy pages

2. Native Text: Extracts embedded text using PyMuPDF with positional bounding boxes to preserve layout and reading order

3. Image Processing: Identifies embedded images, filters out decorative elements (<50x50px, <1KB), extracts position data for proper ordering

4. Image Enhancement: Applies multiple preprocessing techniques - grayscale conversion, noise reduction, OTSU/adaptive thresholding, contrast enhancement

5. Multi-Method OCR: Runs Tesseract with various PSM modes (6,7,8,11,13), tries preprocessed images, scales images 2x for small text, selects best result

6. Position Integration: Sorts all content (text + images) by vertical position to maintain top-to-bottom, left-to-right reading flow

7. Fallback OCR: Converts entire PDF pages to 300 DPI images for full-page OCR when native text extraction is insufficient

8. Output Generation: Combines all extraction methods into structured text file with detailed extraction report and source identification


  
## Output Format  

(Creates a txt file)  

PDF TEXT EXTRACTION REPORT  
Source PDF: sample.pdf  
Total pages: 3  
Images Found 8,421  
  

## Requirements
numpy==2.2.6  
opencv-python==4.12.0.88  
packaging==25.0  
pdf2image==1.17.0  
pillow==11.3.0  
PyMuPDF==1.26.3  
pytesseract==0.3.13
