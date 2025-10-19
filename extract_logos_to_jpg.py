"""
Extract all logos from cache and save as JPG files
Creates a folder with all extracted logos organized by domain
"""

import pickle
import numpy as np
import cv2
from PIL import Image
import io
import os
import re
from urllib.parse import urlparse
from datetime import datetime

def sanitize_filename(filename):
    """Sanitize filename for safe file system storage"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    # Ensure it doesn't start/end with dots or spaces
    filename = filename.strip('. ')
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def extract_domain_name(domain_or_url):
    """Extract clean domain name from URL or domain string"""
    try:
        if domain_or_url.startswith(('http://', 'https://')):
            parsed = urlparse(domain_or_url)
            domain = parsed.netloc
        else:
            domain = domain_or_url
        
        # Remove www. prefix
        domain = domain.replace('www.', '')
        
        # Take the main domain part (before first dot for brand name)
        parts = domain.split('.')
        if len(parts) > 1:
            brand_name = parts[0]
        else:
            brand_name = domain
            
        return sanitize_filename(brand_name)
    except:
        return sanitize_filename(domain_or_url)

def convert_bytes_to_jpg(logo_bytes, output_path, domain_name):
    """Convert logo bytes to JPG file with multiple fallback strategies"""
    try:
        # Convert bytes to numpy array
        img_array = np.frombuffer(logo_bytes, dtype=np.uint8)
        
        # Strategy 1: Direct color decode with OpenCV
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Strategy 2: If failed, try grayscale then convert
        if img is None:
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Strategy 3: Try with any flags
        if img is None:
            img = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is not None and len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img is not None and img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Strategy 4: Try PIL as fallback
        if img is None:
            try:
                pil_img = Image.open(io.BytesIO(logo_bytes))
                # Convert to RGB if needed
                if pil_img.mode == 'RGBA':
                    # Create white background for transparency
                    background = Image.new('RGB', pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[-1] if pil_img.mode == 'RGBA' else None)
                    pil_img = background
                elif pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # Convert PIL to OpenCV format
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as pil_error:
                print(f"PIL fallback failed for {domain_name}: {pil_error}")
                return False
        
        # Validate image
        if img is not None and img.shape[0] > 10 and img.shape[1] > 10:
            # Ensure 3 channels (BGR for OpenCV)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Save as JPG with high quality
            success = cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return success
        else:
            print(f"Invalid image dimensions for {domain_name}: {img.shape if img is not None else 'None'}")
            return False
            
    except Exception as e:
        print(f"Error converting {domain_name}: {e}")
        return False

def extract_all_logos():
    """Extract all logos from cache and save as JPG files"""
    
    print("="*60)
    print("LOGO EXTRACTION TO JPG FILES")
    print("="*60)
    
    # Load cached logo data
    cache_path = "comprehensive_logo_extraction_fast_results.pkl"
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file {cache_path} not found!")
        return
    
    print(f"Loading logos from {cache_path}...")
    
    with open(cache_path, 'rb') as f:
        cached_data = pickle.load(f)
    
    # Use successful_logos for better success rate
    logo_results = cached_data['successful_logos']
    print(f"Found {len(logo_results)} successful logo entries")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"extracted_logos_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    
    # Extract and save logos
    successful_saves = 0
    failed_saves = 0
    
    for i, logo_entry in enumerate(logo_results):
        if (i + 1) % 100 == 0:
            progress = ((i + 1) / len(logo_results)) * 100
            print(f"Progress: {i + 1}/{len(logo_results)} ({progress:.1f}%)")
        
        try:
            # Get domain and logo data
            domain = logo_entry.get('domain', logo_entry.get('website', f'unknown_{i}'))
            logo_bytes = logo_entry.get('logo_data')
            
            if not logo_bytes:
                failed_saves += 1
                continue
            
            # Create safe filename
            domain_name = extract_domain_name(domain)
            
            # Handle duplicates by adding counter
            base_filename = f"{domain_name}.jpg"
            output_path = os.path.join(output_dir, base_filename)
            
            counter = 1
            while os.path.exists(output_path):
                filename_with_counter = f"{domain_name}_{counter}.jpg"
                output_path = os.path.join(output_dir, filename_with_counter)
                counter += 1
            
            # Convert and save
            success = convert_bytes_to_jpg(logo_bytes, output_path, domain_name)
            
            if success:
                successful_saves += 1
            else:
                failed_saves += 1
                
        except Exception as e:
            print(f"Error processing logo {i}: {e}")
            failed_saves += 1
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    
    total_logos = len(logo_results)
    success_rate = (successful_saves / total_logos) * 100 if total_logos > 0 else 0
    
    print(f" Total logos processed: {total_logos}")
    print(f" Successfully saved: {successful_saves}")
    print(f" Failed to save: {failed_saves}")
    print(f" Success rate: {success_rate:.2f}%")
    print(f" Output directory: {output_dir}")
    
    # Show directory info
    if os.path.exists(output_dir):
        jpg_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        total_size_mb = sum(os.path.getsize(os.path.join(output_dir, f)) for f in jpg_files) / (1024 * 1024)
        
        print(f" JPG files created: {len(jpg_files)}")
        print(f" Total folder size: {total_size_mb:.1f} MB")
    
    # List sample files
    if successful_saves > 0:
        sample_files = os.listdir(output_dir)[:10]
        print(f"\nSample files:")
        for filename in sample_files:
            print(f"  {filename}")
        if len(os.listdir(output_dir)) > 10:
            print(f"  ... and {len(os.listdir(output_dir)) - 10} more")
    
    print(f"\n Logo extraction completed! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    extract_all_logos()
