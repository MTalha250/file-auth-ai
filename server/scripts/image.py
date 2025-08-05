import cv2
import numpy as np
import os
import argparse
import json
import requests
from PIL import Image
from io import BytesIO
from typing import Dict
from urllib.parse import urlparse

class EasyOCRExtractor:
    def __init__(self, enable_gpu: bool = True):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=enable_gpu)
        except ImportError:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
    
    def is_url(self, path: str) -> bool:
        """Check if the path is a URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def load_image_from_url(self, url: str) -> np.ndarray:
        """Load image directly from URL into memory"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Convert to PIL Image
            pil_image = Image.open(BytesIO(response.content))
            
            # Convert PIL to OpenCV format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # PIL uses RGB, OpenCV uses BGR
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
        except requests.RequestException as e:
            raise ValueError(f"Failed to download image from URL: {e}")
        except Exception as e:
            raise ValueError(f"Failed to process image: {e}")
    
    def preprocess_image(self, image_input) -> np.ndarray:
        """
        Preprocess image from either file path or URL
        Args:
            image_input: Either a file path (str) or numpy array
        """
        if isinstance(image_input, str):
            if self.is_url(image_input):
                image = self.load_image_from_url(image_input)
            else:
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image: {image_input}")
        else:
            image = image_input
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        height, width = enhanced.shape[:2]
        if width < 500 or height < 500:
            scale = max(500 / width, 500 / height)
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return enhanced
    
    def extract_text(self, image_path: str) -> Dict:
        try:
            if self.is_url(image_path):
                # Load image from URL
                image_array = self.load_image_from_url(image_path)
                # EasyOCR can work with numpy arrays directly
                results = self.reader.readtext(image_array)
                source_type = 'url'
            else:
                if not os.path.exists(image_path):
                    return {'text': '', 'confidence': 0, 'error': f"Image not found: {image_path}"}
                results = self.reader.readtext(image_path)
                source_type = 'local'
            
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.1:
                    texts.append(text.strip())
                    confidences.append(confidence)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': '\n'.join(texts),
                'confidence': avg_confidence,
                'word_count': len(texts),
                'source': source_type
            }
        except Exception as e:
            return {'text': '', 'confidence': 0, 'error': str(e)}

def process_image_file(image_path: str, output_dir: str = "extracted_texts", save_result: bool = True) -> Dict[str, Any]:
    """
    Process an image file and extract text using EasyOCR.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save extracted text files
        save_result: Whether to save the result to a file
        
    Returns:
        Dictionary containing extraction results and output file path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        extractor = EasyOCRExtractor(enable_gpu=True)
        result = extractor.extract_text(image_path)
        
        print(f"Image text extraction completed!")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Words extracted: {result.get('word_count', 0)}")
        print(f"Characters extracted: {len(result.get('text', ''))}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return result
        
        # Save result to file if requested
        if save_result and result.get('text', '').strip():
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(exist_ok=True)
            
            image_name = Path(image_path).stem
            output_filename = f"{image_name}_extracted_text.txt"
            output_path = output_dir_path / output_filename
            
            # Create summary header
            summary = [
                f"IMAGE TEXT EXTRACTION REPORT",
                f"Source Image: {image_path}",
                f"Confidence: {result['confidence']:.2f}",
                f"Words Extracted: {result.get('word_count', 0)}",
                f"Total Characters: {len(result.get('text', ''))}",
                "="*80,
                ""
            ]
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary))
                f.write(result.get('text', ''))
            
            result['output_file'] = str(output_path)
            print(f"Results saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='EasyOCR Text Extractor with URL support (in-memory)')
    parser.add_argument('image_path', help='Path to image file or URL')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--save-json', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        extractor = EasyOCRExtractor(enable_gpu=not args.no_gpu)
        result = extractor.extract_text(args.image_path)
        
        print(f"Source: {result.get('source', 'unknown')}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Words extracted: {result.get('word_count', 0)}")
        print("\nExtracted Text:")
        print(result['text'])
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return 1
        
        if args.save_json:
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.save_json}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())