import fitz  # PyMuPDF
from PIL import Image
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import io

# Import EasyOCR extractor from image.py
from image import EasyOCRExtractor


class PDFTextExtractor:
    """
    A robust PDF text extraction pipeline that combines native text extraction
    with OCR-based image text detection.
    """
    
    def __init__(self, output_dir: str = "extracted_texts", use_easyocr: bool = True):
        """
        Initialize the PDF text extractor.
        
        Args:
            output_dir: Directory to save extracted text files
            use_easyocr: Whether to use EasyOCR via imported image.py class
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.temp_images_dir = self.output_dir / "temp_images"
        self.temp_images_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.use_easyocr = use_easyocr
        self.easyocr_extractor = None
        
        if self.use_easyocr:
            try:
                self.logger.info("Initializing EasyOCR from image.py...")
                self.easyocr_extractor = EasyOCRExtractor(enable_gpu=True)
                self.logger.info("EasyOCR initialized successfully with GPU")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EasyOCR with GPU, trying CPU: {e}")
                try:
                    self.easyocr_extractor = EasyOCRExtractor(enable_gpu=False)
                    self.logger.info("EasyOCR initialized successfully with CPU")
                except Exception as e2:
                    self.logger.error(f"Failed to initialize EasyOCR: {e2}")
                    self.use_easyocr = False
    
    def diagnose_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Diagnose PDF structure to understand content types.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with diagnostic information
        """
        try:
            pdf_document = fitz.open(pdf_path)
            diagnosis = {
                'total_pages': len(pdf_document),
                'page_details': []
            }
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Get basic page info
                page_info = {
                    'page_num': page_num + 1,
                    'mediabox': page.mediabox,
                    'rotation': page.rotation,
                    'image_count': len(page.get_images(full=True)),
                    'text_length': len(page.get_text()),
                    'drawings_count': 0,  # Initialize to 0
                    'links_count': len(page.get_links()),
                    'annotations_count': 0  # Initialize to 0
                }
                
                # Safely get drawings count
                try:
                    drawings = page.get_drawings()
                    page_info['drawings_count'] = len(list(drawings))  # Convert generator to list
                except:
                    page_info['drawings_count'] = 0
                
                # Safely get annotations count
                try:
                    annotations = page.annots()
                    page_info['annotations_count'] = len(list(annotations))  # Convert generator to list
                except:
                    page_info['annotations_count'] = 0
                
                # Check for different content types
                images = page.get_images(full=True)
                page_info['image_details'] = []
                
                for img_idx, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        img_detail = {
                            'index': img_idx + 1,
                            'xref': xref,
                            'ext': base_image.get('ext', 'unknown'),
                            'width': base_image.get('width', 0),
                            'height': base_image.get('height', 0),
                            'colorspace': base_image.get('colorspace', 'unknown'),
                            'size_bytes': len(base_image.get('image', b''))
                        }
                        page_info['image_details'].append(img_detail)
                    except Exception as e:
                        page_info['image_details'].append({
                            'index': img_idx + 1,
                            'error': str(e)
                        })
                
                diagnosis['page_details'].append(page_info)
                
                self.logger.info(f"Page {page_num + 1}: {page_info['text_length']} chars text, "
                               f"{page_info['image_count']} images, {page_info['drawings_count']} drawings")
            
            pdf_document.close()
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"PDF diagnosis failed: {e}")
            return {'error': str(e)}
    
    def extract_native_text(self, pdf_document: fitz.Document) -> List[Dict]:
        """
        Extract native text from PDF pages using PyMuPDF with positional information.
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            List of dictionaries containing text content and position for each page
        """
        page_texts = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            text_dict = page.get_text("dict")
            simple_text = page.get_text("text")
            
            text_blocks = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                        block_text += "\n"
                    
                    if block_text.strip():
                        text_blocks.append({
                            "text": block_text.strip(),
                            "bbox": block["bbox"],
                            "type": "text"
                        })
            
            page_data = {
                "page_num": page_num,
                "simple_text": simple_text,
                "text_blocks": text_blocks,
                "page_height": page.rect.height,
                "page_width": page.rect.width
            }
            
            page_texts.append(page_data)
            self.logger.info(f"Extracted native text from page {page_num + 1} with {len(text_blocks)} text blocks")
        
        return page_texts
    
    def extract_images_from_pdf(self, pdf_document: fitz.Document) -> List[Tuple[int, List[Dict]]]:
        """
        Enhanced image extraction from PDF pages with position information.
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            List of tuples containing (page_number, list_of_image_dicts_with_position)
        """
        all_images = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_images = []
            
            image_list = page.get_images(full=True)
            self.logger.info(f"Page {page_num + 1}: Found {len(image_list)} images using get_images()")
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    if len(image_bytes) < 1000:
                        self.logger.debug(f"Skipping tiny image {img_index + 1} on page {page_num + 1} (size: {len(image_bytes)} bytes)")
                        continue
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    if image.width < 50 or image.height < 50:
                        self.logger.debug(f"Skipping small image {img_index + 1} on page {page_num + 1} (dimensions: {image.width}x{image.height})")
                        continue
                    
                    image_filename = f"page_{page_num + 1}_image_{img_index + 1}.png"
                    image_path = self.temp_images_dir / image_filename
                    image.save(image_path, "PNG")
                    self.logger.info(f"Saved image to: {image_path}")
                    
                    img_bbox = None
                    try:
                        for item in page.get_images(full=True):
                            if item[0] == xref:
                                img_rects = page.get_image_rects(item)
                                if img_rects:
                                    img_bbox = img_rects[0]
                                break
                    except:
                        img_bbox = fitz.Rect(0, 0, image.width, image.height)
                    
                    image_data = {
                        "image_path": str(image_path),
                        "bbox": img_bbox,
                        "index": img_index + 1,
                        "size": (image.width, image.height),
                        "type": "image",
                        "filename": image_filename
                    }
                    
                    page_images.append(image_data)
                    self.logger.info(f"Successfully extracted image {img_index + 1} from page {page_num + 1} (size: {image.width}x{image.height}, {len(image_bytes)} bytes)")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract image {img_index + 1} from page {page_num + 1}: {e}")
            
            all_images.append((page_num, page_images))
            self.logger.info(f"Page {page_num + 1}: Successfully extracted {len(page_images)} valid images")
        
        return all_images
    
    def extract_text_using_easyocr(self, image_path: str) -> Dict[str, any]:
        """
        Extract text from image using imported EasyOCRExtractor from image.py.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing OCR results
        """
        if not self.use_easyocr or self.easyocr_extractor is None:
            return {'text': '', 'confidence': 0, 'error': 'EasyOCR not available'}
        
        try:
            # Use the exact same method from your image.py
            result = self.easyocr_extractor.extract_text(str(image_path))
            
            self.logger.info(f"EasyOCR extracted {len(result.get('text', ''))} characters from {Path(image_path).name} (confidence: {result.get('confidence', 0):.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"EasyOCR failed for image {image_path}: {e}")
            return {'text': '', 'confidence': 0, 'error': str(e)}
    
    def combine_text_and_images_by_position(self, page_text_data: Dict, page_images: List[Dict]) -> List[Dict]:
        """
        Combine text blocks and images based on their position on the page to maintain reading order.
        
        Args:
            page_text_data: Dictionary containing text blocks and page info
            page_images: List of image dictionaries with position info
            
        Returns:
            List of content blocks sorted by reading order (top to bottom, left to right)
        """
        content_blocks = []
        
        for text_block in page_text_data.get("text_blocks", []):
            content_blocks.append({
                "type": "text",
                "content": text_block["text"],
                "bbox": text_block["bbox"],
                "y_position": text_block["bbox"][1]
            })
        
        for img_data in page_images:
            bbox = img_data.get("bbox")
            if bbox:
                y_pos = bbox.y0 if hasattr(bbox, 'y0') else bbox[1] if isinstance(bbox, (list, tuple)) else 0
            else:
                y_pos = 0
            
            content_blocks.append({
                "type": "image",
                "content": img_data,
                "bbox": bbox,
                "y_position": y_pos
            })
        
        content_blocks.sort(key=lambda x: x["y_position"])
        
        return content_blocks
    
    def extract_text_from_pdf_as_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to images and extract text using OCR.
        This is useful for scanned PDFs or when native text extraction fails.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text content for each page
        """
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path, dpi=300)
            page_texts = []
            
            for i, image in enumerate(images):
                text = ""
                page_texts.append(text)
                self.logger.info(f"Fallback OCR for PDF page {i + 1} (as image) - currently disabled")
            
            return page_texts
            
        except Exception as e:
            self.logger.error(f"Failed to convert PDF to images: {e}")
            return []
    
    def process_pdf(self, pdf_path: str, use_fallback_ocr: bool = True, diagnose: bool = True) -> Dict[str, any]:
        """
        Main method to process a PDF file and extract all text content.
        
        Args:
            pdf_path: Path to the PDF file
            use_fallback_ocr: Whether to use full-page OCR as fallback
            diagnose: Whether to run PDF diagnostics first
            
        Returns:
            Dictionary containing extraction results
        """
        self.logger.info(f"Starting PDF processing: {pdf_path}")
        
        results = {
            'pdf_path': pdf_path,
            'native_text': [],
            'image_ocr_text': [],
            'fallback_ocr_text': [],
            'combined_text': '',
            'total_pages': 0,
            'images_found': 0,
            'diagnosis': None
        }
        
        try:
            if diagnose:
                self.logger.info("Running PDF diagnostics...")
                diagnosis = self.diagnose_pdf(pdf_path)
                results['diagnosis'] = diagnosis
                
                total_images = sum(page['image_count'] for page in diagnosis.get('page_details', []))
                self.logger.info(f"Diagnosis complete: {diagnosis.get('total_pages', 0)} pages, {total_images} total images detected")
            
            pdf_document = fitz.open(pdf_path)
            results['total_pages'] = len(pdf_document)
            
            self.logger.info("Extracting native text...")
            native_texts = self.extract_native_text(pdf_document)
            results['native_text'] = native_texts
            
            self.logger.info("Extracting images and performing OCR...")
            page_images = self.extract_images_from_pdf(pdf_document)
            
            ordered_content = []
            total_images = 0
            
            for page_num, images in page_images:
                page_image_count = len(images)
                total_images += page_image_count
                
                self.logger.info(f"Processing {page_image_count} images from page {page_num + 1}")
                
                processed_images = []
                for img_idx, image_data in enumerate(images):
                    try:
                        image_path = image_data["image_path"]
                        self.logger.info(f"Starting EasyOCR for image {img_idx + 1} on page {page_num + 1} at {image_path}")
                        
                        if self.use_easyocr:
                            ocr_result = self.extract_text_using_easyocr(image_path)
                            ocr_text = ocr_result.get('text', '')
                            confidence = ocr_result.get('confidence', 0)
                        else:
                            ocr_text = ""
                            confidence = 0
                        
                        image_data["ocr_text"] = ocr_text
                        image_data["ocr_confidence"] = confidence
                        processed_images.append(image_data)
                        
                        if ocr_text.strip():
                            self.logger.info(f"EasyOCR extracted {len(ocr_text)} characters from image {img_idx + 1} on page {page_num + 1} (confidence: {confidence:.2f})")
                        else:
                            self.logger.warning(f"No text found in image {img_idx + 1} on page {page_num + 1}")
                            
                    except Exception as e:
                        self.logger.error(f"EasyOCR failed for image {img_idx + 1} on page {page_num + 1}: {e}")
                        image_data["ocr_text"] = ""
                        image_data["ocr_confidence"] = 0
                        processed_images.append(image_data)
                
                if page_num < len(native_texts):
                    page_content = self.combine_text_and_images_by_position(
                        native_texts[page_num], processed_images
                    )
                    ordered_content.append(page_content)
                else:
                    ordered_content.append([{
                        "type": "image",
                        "content": img_data,
                        "bbox": img_data.get("bbox"),
                        "y_position": 0
                    } for img_data in processed_images])
            
            results['ordered_content'] = ordered_content
            results['images_found'] = total_images
            
            pdf_document.close()
            
            if use_fallback_ocr:
                total_native_chars = 0
                for page_data in native_texts:
                    if isinstance(page_data, dict):
                        total_native_chars += len(page_data.get('simple_text', ''))
                    else:
                        total_native_chars += len(str(page_data))
                
                if total_native_chars < 100:
                    self.logger.info("Native text extraction yielded minimal results. Using full-page OCR...")
                    fallback_texts = self.extract_text_from_pdf_as_images(pdf_path)
                    results['fallback_ocr_text'] = fallback_texts
            
            combined_text = self._combine_extracted_text(results)
            results['combined_text'] = combined_text
            
            self.logger.info(f"PDF processing completed. Total characters extracted: {len(combined_text)}")
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise
        
        return results
    
    def _combine_extracted_text(self, results: Dict[str, any]) -> str:
        """
        Combine all extracted text into a single string, maintaining reading order.
        
        Args:
            results: Dictionary containing extraction results
            
        Returns:
            Combined text string with preserved reading order
        """
        combined_parts = []
        
        if 'ordered_content' in results and results['ordered_content']:
            for page_num, page_content in enumerate(results['ordered_content']):
                if not page_content:
                    continue
                    
                combined_parts.append(f"\n=== PAGE {page_num + 1} ===")
                
                for block in page_content:
                    if block["type"] == "text":
                        combined_parts.append(block["content"])
                    elif block["type"] == "image":
                        img_data = block["content"]
                        ocr_text = img_data.get("ocr_text", "")
                        confidence = img_data.get("ocr_confidence", 0)
                        image_index = img_data.get('index', '?')
                        
                        if ocr_text.strip():
                            combined_parts.append(f"\n[IMAGE {image_index} - EasyOCR Confidence: {confidence:.2f}]")
                            combined_parts.append(ocr_text)
                            combined_parts.append(f"[IMAGE {image_index} END]")
                        else:
                            combined_parts.append(f"\n[IMAGE {image_index} - No text detected by EasyOCR]")
                
                combined_parts.append("\n" + "="*50 + "\n")
        
        else:
            for page_num in range(results['total_pages']):
                page_parts = []
                
                if page_num < len(results.get('native_text', [])):
                    if isinstance(results['native_text'][page_num], dict):
                        simple_text = results['native_text'][page_num].get('simple_text', '')
                        if simple_text.strip():
                            page_parts.append("=== NATIVE TEXT ===")
                            page_parts.append(simple_text)
                    elif isinstance(results['native_text'][page_num], str):
                        if results['native_text'][page_num].strip():
                            page_parts.append("=== NATIVE TEXT ===")
                            page_parts.append(results['native_text'][page_num])
                
                if (results.get('image_ocr_text') and 
                    page_num < len(results['image_ocr_text']) and 
                    results['image_ocr_text'][page_num].strip()):
                    page_parts.append("=== IMAGE OCR TEXT ===")
                    page_parts.append(results['image_ocr_text'][page_num])
                
                if (results.get('fallback_ocr_text') and 
                    page_num < len(results['fallback_ocr_text']) and 
                    results['fallback_ocr_text'][page_num].strip()):
                    page_parts.append("=== FALLBACK OCR TEXT ===")
                    page_parts.append(results['fallback_ocr_text'][page_num])
                
                if page_parts:
                    combined_parts.append(f"\n=== PAGE {page_num + 1} ===")
                    combined_parts.extend(page_parts)
                    combined_parts.append("\n" + "="*50 + "\n")
        
        return '\n'.join(combined_parts)
    
    def save_extracted_text(self, results: Dict[str, any], output_filename: Optional[str] = None) -> str:
        """
        Save the extracted text to a file.
        
        Args:
            results: Dictionary containing extraction results
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to the saved file
        """
        if not output_filename:
            pdf_name = Path(results['pdf_path']).stem
            output_filename = f"{pdf_name}_extracted_text.txt"
        
        output_path = self.output_dir / output_filename
        
        summary = [
            f"PDF TEXT EXTRACTION REPORT",
            f"Source PDF: {results['pdf_path']}",
            f"Total Pages: {results['total_pages']}",
            f"Images Found: {results['images_found']}",
            f"Total Characters: {len(results['combined_text'])}",
            f"Extraction Date: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}",
            "="*80,
            ""
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
            f.write(results['combined_text'])
        
        self.logger.info(f"Extracted text saved to: {output_path}")
        return str(output_path)
    
    def cleanup_temp_files(self):
        """
        Clean up temporary image files.
        """
        try:
            if self.temp_images_dir.exists():
                import shutil
                shutil.rmtree(self.temp_images_dir)
                self.logger.info(f"Cleaned up temporary images directory: {self.temp_images_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temporary files: {e}")


def process_pdf_file(pdf_path: str, output_dir: str = "extracted_texts", use_easyocr: bool = True, use_fallback_ocr: bool = True) -> Dict[str, Any]:
    """
    Process a PDF file and extract all text content.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted text files
        use_easyocr: Whether to use EasyOCR for image text extraction
        use_fallback_ocr: Whether to use full-page OCR as fallback
        
    Returns:
        Dictionary containing extraction results and output file path
    """
    # Initialize the extractor with EasyOCR support (imported from image.py)
    extractor = PDFTextExtractor(
        output_dir=output_dir,
        use_easyocr=use_easyocr
    )
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Extract text from PDF using hybrid approach (native text + EasyOCR)
        results = extractor.process_pdf(pdf_path, use_fallback_ocr=use_fallback_ocr, diagnose=True)
        
        # Save extracted text
        output_file = extractor.save_extracted_text(results)
        results['output_file'] = output_file
        
        print(f"PDF text extraction completed!")
        print(f"Results saved to: {output_file}")
        print(f"Total pages processed: {results['total_pages']}")
        print(f"Images found and processed: {results['images_found']}")
        print(f"Total characters extracted: {len(results['combined_text'])}")
        
        # Clean up temporary images
        extractor.cleanup_temp_files()
        print(f"Cleaned up temporary files")
        
        return results
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """
    Example usage of the PDF text extraction pipeline.
    """
    pdf_path = "sample2.pdf"  # Replace with your PDF path
    
    try:
        results = process_pdf_file(pdf_path)
        print("PDF processing completed successfully!")
    except Exception as e:
        print(f"PDF processing failed: {e}")


if __name__ == "__main__":
    main()
