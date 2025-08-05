import cv2
import numpy as np
import os
import argparse
import json
from typing import Dict, Any
from pathlib import Path

class EasyOCRExtractor:
    def __init__(self, enable_gpu: bool = True):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=enable_gpu)
        except ImportError:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        height, width = enhanced.shape[:2]
        if width < 500 or height < 500:
            scale = max(500 / width, 500 / height)
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return enhanced
    
    def extract_text(self, image_path: str) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            results = self.reader.readtext(image_path)
            
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
                'word_count': len(texts)
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
    parser = argparse.ArgumentParser(description='EasyOCR Text Extractor')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--save-json', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        extractor = EasyOCRExtractor(enable_gpu=not args.no_gpu)
        result = extractor.extract_text(args.image_path)
        
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Words extracted: {result.get('word_count', 0)}")
        print("\nExtracted Text:")
        print(result['text'])
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        
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