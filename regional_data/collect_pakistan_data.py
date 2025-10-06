"""
Pakistan/South Asian Regional Data Collection Pipeline
Collects region-specific object images using web scraping and APIs.

Features:
- Google Images scraping for Pakistan-specific objects
- Bing Images API integration
- Automatic image filtering and validation
- Duplicate detection and removal
- Quality assessment
"""

import os
import requests
import json
import time
import hashlib
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import logging
from urllib.parse import urlparse, urljoin
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PakistanDataCollector:
    def __init__(self, output_dir="regional_data/pakistan"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pakistan-specific object categories
        self.pakistan_categories = {
            # Currency
            "currency": [
                "pakistani rupee notes", "pakistan currency", "rupee coins",
                "1000 rupee note", "500 rupee note", "100 rupee note", "50 rupee note",
                "20 rupee note", "10 rupee note", "5 rupee note", "1 rupee coin", "2 rupee coin"
            ],
            
            # Food items
            "food": [
                "roti chapati", "naan bread", "biryani rice", "karahi curry",
                "chai tea cup", "lassi drink", "samosa", "pakora", "kebab",
                "daal lentils", "sabzi vegetables", "halwa dessert", "kulfi ice cream",
                "paratha bread", "qorma curry", "nihari stew", "haleem dish"
            ],
            
            # Clothing
            "clothing": [
                "shalwar kameez", "dupatta scarf", "kurta shirt", "sherwani",
                "lehenga dress", "churidar pants", "waistcoat", "pagri turban",
                "chador hijab", "burqa", "topi cap", "khussay shoes", "chappal sandals"
            ],
            
            # Transportation
            "transportation": [
                "rickshaw auto", "chingchi rickshaw", "suzuki van", "local bus pakistan",
                "truck art pakistan", "motorcycle 70cc", "bicycle pakistan",
                "tanga horse cart", "donkey cart", "camel cart"
            ],
            
            # Household items
            "household": [
                "hookah shisha", "surahi water pot", "tawa griddle", "degchi pot",
                "charpoy bed", "takht platform", "pankha fan", "matka water pot",
                "thali plate", "lota water vessel", "chimta tongs", "belan rolling pin"
            ],
            
            # Religious items
            "religious": [
                "prayer mat janamaz", "tasbih beads", "quran holy book", "mosque minaret",
                "mihrab prayer niche", "eid decorations", "crescent moon star",
                "islamic calligraphy", "prayer cap", "athan clock", "qibla compass"
            ],
            
            # Street/Market items
            "street_market": [
                "fruit cart", "vegetable vendor", "street food stall", "paan shop",
                "tea stall", "barber shop pole", "tailor shop", "shoe repair",
                "mobile shop", "general store", "bakery tandoor", "meat shop"
            ],
            
            # Cultural items
            "cultural": [
                "truck art", "henna mehndi", "bangles chooriyan", "nose ring nath",
                "earrings jhumka", "traditional jewelry", "embroidery work",
                "mirror work", "block printing", "ajrak cloth", "sindhi topi"
            ]
        }
        
        # Image quality thresholds
        self.min_image_size = (224, 224)
        self.max_image_size = (2048, 2048)
        self.min_file_size = 10 * 1024  # 10KB
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
        # Duplicate detection
        self.image_hashes = set()
        
    def setup_webdriver(self, headless=True):
        """Setup Chrome webdriver for web scraping"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup webdriver: {e}")
            logger.info("Please install ChromeDriver: https://chromedriver.chromium.org/")
            return None
    
    def calculate_image_hash(self, image_path):
        """Calculate perceptual hash for duplicate detection"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Resize to 8x8 and convert to grayscale
            img = cv2.resize(img, (8, 8))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate average
            avg = img.mean()
            
            # Create hash
            hash_bits = []
            for pixel in img.flatten():
                hash_bits.append('1' if pixel > avg else '0')
            
            return ''.join(hash_bits)
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {image_path}: {e}")
            return None
    
    def is_valid_image(self, image_path):
        """Validate image quality and format"""
        try:
            with Image.open(image_path) as img:
                # Check format
                if img.format not in ['JPEG', 'PNG', 'JPG']:
                    return False, "Invalid format"
                
                # Check size
                if img.size[0] < self.min_image_size[0] or img.size[1] < self.min_image_size[1]:
                    return False, "Too small"
                
                if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                    return False, "Too large"
                
                # Check file size
                file_size = os.path.getsize(image_path)
                if file_size < self.min_file_size or file_size > self.max_file_size:
                    return False, "Invalid file size"
                
                # Check if image is corrupted
                img.verify()
                
                return True, "Valid"
                
        except Exception as e:
            return False, f"Error: {e}"
    
    def is_duplicate(self, image_path):
        """Check if image is duplicate using perceptual hashing"""
        img_hash = self.calculate_image_hash(image_path)
        if img_hash is None:
            return True  # Treat as duplicate if can't calculate hash
        
        if img_hash in self.image_hashes:
            return True
        
        self.image_hashes.add(img_hash)
        return False
    
    def download_image(self, url, save_path, timeout=10):
        """Download image from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, "Not an image"
            
            # Save image
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True, "Downloaded"
            
        except Exception as e:
            return False, f"Download error: {e}"
    
    def scrape_google_images(self, query, max_images=50):
        """Scrape images from Google Images"""
        logger.info(f"Scraping Google Images for: {query}")
        
        driver = self.setup_webdriver()
        if not driver:
            return []
        
        try:
            # Navigate to Google Images
            search_url = f"https://www.google.com/search?q={query}&tbm=isch"
            driver.get(search_url)
            
            # Wait for images to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img[data-src]"))
            )
            
            # Scroll to load more images
            for _ in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Find image elements
            img_elements = driver.find_elements(By.CSS_SELECTOR, "img[data-src], img[src]")
            
            image_urls = []
            for img in img_elements[:max_images]:
                src = img.get_attribute("data-src") or img.get_attribute("src")
                if src and src.startswith("http") and "base64" not in src:
                    image_urls.append(src)
            
            logger.info(f"Found {len(image_urls)} image URLs for {query}")
            return image_urls
            
        except Exception as e:
            logger.error(f"Error scraping Google Images for {query}: {e}")
            return []
        
        finally:
            driver.quit()
    
    def collect_category_images(self, category, queries, max_per_query=30):
        """Collect images for a specific category"""
        logger.info(f"Collecting images for category: {category}")
        
        category_dir = self.output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        collected_count = 0
        total_downloaded = 0
        
        for query in tqdm(queries, desc=f"Processing {category}"):
            query_dir = category_dir / query.replace(" ", "_").replace("/", "_")
            query_dir.mkdir(exist_ok=True)
            
            # Scrape image URLs
            image_urls = self.scrape_google_images(query, max_per_query)
            
            # Download images
            for i, url in enumerate(image_urls):
                if collected_count >= max_per_query * len(queries):
                    break
                
                try:
                    # Generate filename
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filename = f"{query.replace(' ', '_')}_{i:03d}_{url_hash}.jpg"
                    save_path = query_dir / filename
                    
                    # Skip if already exists
                    if save_path.exists():
                        continue
                    
                    # Download image
                    success, message = self.download_image(url, save_path)
                    if not success:
                        continue
                    
                    # Validate image
                    valid, validation_message = self.is_valid_image(save_path)
                    if not valid:
                        save_path.unlink()
                        continue
                    
                    # Check for duplicates
                    if self.is_duplicate(save_path):
                        save_path.unlink()
                        continue
                    
                    collected_count += 1
                    total_downloaded += 1
                    
                    # Small delay to be respectful
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error processing {url}: {e}")
                    continue
        
        logger.info(f"Collected {collected_count} images for {category}")
        return collected_count
    
    def collect_all_categories(self, max_per_query=30):
        """Collect images for all Pakistan-specific categories"""
        logger.info("Starting Pakistan regional data collection...")
        
        total_collected = 0
        collection_stats = {}
        
        for category, queries in self.pakistan_categories.items():
            try:
                count = self.collect_category_images(category, queries, max_per_query)
                collection_stats[category] = count
                total_collected += count
                
                # Save progress
                self.save_collection_stats(collection_stats, total_collected)
                
                # Delay between categories
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error collecting {category}: {e}")
                collection_stats[category] = 0
        
        logger.info(f"Collection completed! Total images: {total_collected}")
        return collection_stats
    
    def save_collection_stats(self, stats, total):
        """Save collection statistics"""
        stats_data = {
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': total,
            'categories': stats,
            'pakistan_categories': list(self.pakistan_categories.keys())
        }
        
        stats_path = self.output_dir / 'collection_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"Collection stats saved: {stats_path}")
    
    def create_pakistan_classes_file(self):
        """Create classes.txt file for Pakistan-specific objects"""
        classes = []
        
        for category, queries in self.pakistan_categories.items():
            for query in queries:
                # Normalize class name
                class_name = query.lower().replace(" ", "_").replace("-", "_")
                classes.append(class_name)
        
        classes_path = self.output_dir / 'pakistan_classes.txt'
        with open(classes_path, 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        logger.info(f"Pakistan classes file created: {classes_path}")
        logger.info(f"Total Pakistan-specific classes: {len(classes)}")
        
        return classes_path, len(classes)
    
    def cleanup_invalid_images(self):
        """Clean up invalid or corrupted images"""
        logger.info("Cleaning up invalid images...")
        
        removed_count = 0
        
        for image_path in self.output_dir.rglob("*.jpg"):
            valid, message = self.is_valid_image(image_path)
            if not valid:
                try:
                    image_path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed invalid image: {image_path} ({message})")
                except Exception as e:
                    logger.warning(f"Failed to remove {image_path}: {e}")
        
        logger.info(f"Cleanup completed. Removed {removed_count} invalid images.")
        return removed_count

def main():
    """Main function to run Pakistan data collection"""
    collector = PakistanDataCollector()
    
    print("🇵🇰 Starting Pakistan Regional Data Collection")
    print("=" * 50)
    
    try:
        # Collect images for all categories
        stats = collector.collect_all_categories(max_per_query=20)
        
        # Create classes file
        classes_path, num_classes = collector.create_pakistan_classes_file()
        
        # Cleanup invalid images
        removed = collector.cleanup_invalid_images()
        
        print("\n📊 Collection Summary:")
        print("=" * 30)
        for category, count in stats.items():
            print(f"✅ {category.title()}: {count} images")
        
        print(f"\n🎯 Total Images: {sum(stats.values())}")
        print(f"🏷️  Total Classes: {num_classes}")
        print(f"🧹 Removed Invalid: {removed}")
        print(f"📁 Output Directory: {collector.output_dir}")
        print(f"📋 Classes File: {classes_path}")
        
        print("\n⚠️  Note: This data needs manual annotation or pseudo-labeling")
        print("   Run pseudo_label.py next to create YOLO format labels")
        
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        raise

if __name__ == "__main__":
    main()
