"""
WebScrape - Web Scraping Tool with AI Integration
-----------------------------------------------
This script is a conversion of the Jupyter notebook implementation.
It includes all the functionality from the notebook cells.
"""

# Import Libraries
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from beautifultable import BeautifulTable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import logging
import time
import os
import asyncio
import aiohttp
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
import gc
from psutil import virtual_memory
import warnings
warnings.filterwarnings('ignore')

# Apply nest_asyncio to allow nested event loops (needed for Colab)
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None

# Helper Functions
def load_json(database_json_file="scraped_data.json"):
    """
    This function will load json data from scraped_data.json
    if file exist it will load data from it else creates an empty one"""
    try:
        with open(database_json_file, "r") as read_it:
            all_data_base = json.loads(read_it.read())
            return all_data_base
    except:
        all_data_base = dict()
        return all_data_base

def save_scraped_data_in_json(data, database_json_file="scraped_data.json"):
    """
    This function Save the scraped data in json format. scraped_data.json file if it exist else create it.
    if file already exist you can view previous scraped data in it.
    """
    file_obj = open(database_json_file, "w")
    file_obj.write(json.dumps(data))
    file_obj.close()

def existing_scraped_data_init(json_db):
    """
    This function will check if scraped_data key exist in json_db or not
    """
    scraped_data = json_db.get("scraped_data")
    if scraped_data is None:
        json_db['scraped_data'] = dict()
    return None

def scraped_time_is():
    """
    This function will return current date and time in string format
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

def get_timestamp_filename(prefix, extension):
    """
    Generate a filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

# Web Scraping Functions
def process_url_request(website_url):
    """
    This function will process the url request and return the soup object
    """
    requets_data = requests.get(website_url)
    if requets_data.status_code == 200:
        try:
            soup = BeautifulSoup(requets_data.text, 'lxml')
        except:
            # Fallback to html.parser if lxml is not available
            soup = BeautifulSoup(requets_data.text, 'html.parser')
        return soup
    return None

def proccess_beautiful_soup_data(soup):
    return {
        'title': soup.find('title').text if soup.find('title') else 'No title found',
        'all_anchor_href': [i['href'] for i in soup.find_all('a', href=True)],
        'all_anchors': [str(i) for i in soup.find_all('a')],
        'all_images_data': [str(i) for i in soup.find_all('img')],
        'all_images_source_data': [i.get('src') for i in soup.find_all('img')],
        'all_h1_data': [i.text for i in soup.find_all('h1')],
        'all_h2_data': [i.text for i in soup.find_all('h2')],
        'all_h3_data': [i.text for i in soup.find_all('h3')],
        'all_p_data': [i.text for i in soup.find_all('p')]
    }

# AI Model Setup
def get_colab_gpu_info():
    """Get GPU information for Colab"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")

def clear_memory():
    """Clear memory in Colab environment"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Available RAM: {virtual_memory().available / 1024**3:.2f} GB")

def setup_ai_model(token=None):
    """
    Set up the AI model for content parsing with Colab optimizations
    """
    global model, tokenizer

    try:
        # Check if Hugging Face token is available
        hf_token = token or os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("No Hugging Face token provided. Some models may not be accessible.")

        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        try:
            # Use a more capable model for better parsing
            model_name = "facebook/opt-1.3b"  # Upgraded from 350m to 1.3b
            logger.info(f"Loading model: {model_name}")

            # Load tokenizer with optimized settings
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token if hf_token else None,
                padding_side="left",
                truncation=True,
                model_max_length=2048  # Increased context window
            )

            # Load model with optimized settings
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token if hf_token else None,
                device_map="auto" if device == "cuda" else "cpu",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )

            # Move model to GPU if available
            if device == "cuda":
                model = model.to(device)
                model.eval()

            logger.info("Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.warning(f"Failed to load primary model: {str(e)}")
            try:
                # Fallback to medium model
                logger.info("Trying medium model: facebook/opt-350m")
                model_name = "facebook/opt-350m"

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    padding_side="left",
                    truncation=True,
                    model_max_length=1024
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto" if device == "cuda" else "cpu",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )

                if device == "cuda":
                    model = model.to(device)
                    model.eval()

                logger.info("Medium model loaded successfully")
                return model, tokenizer

            except Exception as e2:
                logger.error(f"Failed to load any model: {str(e2)}")
                return None, None

    except Exception as e:
        logger.error(f"Error setting up AI model: {str(e)}")
        return None, None

# Content Processing Functions
def extract_text_content(data):
    """
    Extract and structure text content from scraped data for AI processing
    """
    text_content = []

    # Structure the content with clear type markers and hierarchy
    if 'title' in data and data['title']:
        text_content.append(f"[PAGE_TITLE]\n{data['title']}\n[/PAGE_TITLE]")

    # Process headings in order of hierarchy with context
    headings = {
        'H1': data.get('all_h1_data', []),
        'H2': data.get('all_h2_data', []),
        'H3': data.get('all_h3_data', [])
    }

    for heading_type, heading_list in headings.items():
        if heading_list:
            text_content.append(f"[{heading_type}_HEADINGS]")
            for heading in heading_list:
                if heading.strip():
                    # Add context for each heading
                    text_content.append(f"<heading>{heading.strip()}</heading>")
            text_content.append(f"[/{heading_type}_HEADINGS]")

    # Process paragraphs with context
    if data.get('all_p_data'):
        text_content.append("[PARAGRAPHS]")
        for p in data['all_p_data']:
            if p.strip():
                # Add context for each paragraph
                text_content.append(f"<paragraph>{p.strip()}</paragraph>")
        text_content.append("[/PARAGRAPHS]")

    # Process links with context
    if data.get('all_anchors'):
        text_content.append("[LINKS]")
        for link in data['all_anchors']:
            if link.strip():
                # Add context for each link
                text_content.append(f"<link>{link.strip()}</link>")
        text_content.append("[/LINKS]")

    # Process images with context
    if data.get('all_images_data'):
        text_content.append("[IMAGES]")
        for img in data['all_images_data']:
            if img.strip():
                # Add context for each image
                text_content.append(f"<image>{img.strip()}</image>")
        text_content.append("[/IMAGES]")

    return "\n".join(text_content)

def chunk_content(content, chunk_size=1500):
    """
    Split content into chunks for processing, with improved chunking strategy
    """
    # First, split by section markers to preserve structure
    sections = re.split(r'(\[.*?\])', content)
    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        if not section.strip():
            continue
            
        # If it's a section marker, add it to current chunk
        if section.startswith('[') and section.endswith(']'):
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(section)
            current_size = len(section)
            continue

        # Split section into sentences while preserving context
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if current_size + len(sentence) + 1 > chunk_size:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence) + 1

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

# AI Parsing Functions
def process_chunk_with_ai(model, tokenizer, chunk, query):
    """
    Process a single chunk with AI for any type of query with improved prompt engineering
    """
    try:
        # Create a more structured and precise prompt based on query type
        base_prompt = f"""You are a precise content extractor. Your task is to extract specific information from the provided content based on the user's query.

QUERY: {query}

CONTENT:
{chunk}

INSTRUCTIONS:
1. Analyze the content carefully and identify all relevant information that matches the query
2. Extract ONLY the specific information requested, maintaining its original context
3. Format the output in a clear, structured way that preserves the information hierarchy
4. Do not add any explanations, labels, or formatting beyond what is necessary
5. If no matching content is found, return "No matching content found"

EXTRACTED CONTENT:
"""

        # Generate response with optimized parameters
        inputs = tokenizer(
            base_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,  # Increased from 512
            padding=True
        )
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,  # Increased from 250
                    do_sample=True,  # Changed to True for better quality
                    num_beams=4,  # Increased from 1
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.5,  # Increased from 1.2
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    temperature=0.7  # Added temperature for better generation
                )
        except Exception as e:
            logger.warning(f"Error with standard generation: {str(e)}")
            return "Error in generation"

        # Get and clean the response
        try:
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Find the position after the prompt
            prompt_end = response.find("EXTRACTED CONTENT:") + len("EXTRACTED CONTENT:")
            if prompt_end == -1:
                return "No matching content found"
                
            response = response[prompt_end:].strip()
            
            # Clean up the response while preserving structure
            response = re.sub(r'<.*?>', '', response)  # Remove HTML tags
            response = re.sub(r'\[.*?\]', '', response)  # Remove markdown-style tags
            response = re.sub(r'^[0-9]+\.|\-|\*|\•', '', response, flags=re.MULTILINE)  # Remove list markers
            
            # Remove any SQL-like queries or command-like text
            response = re.sub(r'SELECT.*?FROM.*?WHERE.*?;?\s*', '', response, flags=re.IGNORECASE|re.MULTILINE)
            response = re.sub(r'INSERT.*?INTO.*?VALUES.*?;?\s*', '', response, flags=re.IGNORECASE|re.MULTILINE)
            response = re.sub(r'UPDATE.*?SET.*?WHERE.*?;?\s*', '', response, flags=re.IGNORECASE|re.MULTILINE)
            response = re.sub(r'DELETE.*?FROM.*?WHERE.*?;?\s*', '', response, flags=re.IGNORECASE|re.MULTILINE)
            response = re.sub(r'#.*$', '', response, flags=re.MULTILINE)
            response = re.sub(r'git.*$', '', response, flags=re.MULTILINE)

            # Final cleaning of lines while preserving meaningful content
            lines = [line.strip() for line in response.split('\n')]
            lines = [line for line in lines if line and 
                    not line.lower().startswith(('here', 'found', 'the ', 'these', 'extracted', 'content', 'results', 
                    'select', 'where', 'from', 'if', 'this', 'please', 'note', 'warning', 'error', 'success',
                    'click', 'read', 'more', 'learn', 'about', 'visit', 'go', 'to', 'see', 'view'))]

            return '\n'.join(lines) if lines else "No matching content found"

        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return "Error processing response"

    except Exception as e:
        logger.error(f"Error in process_chunk_with_ai: {str(e)}")
        return "Error processing chunk"

async def parse_content_with_ai_async(model, tokenizer, content, query):
    """
    Parse content using AI based on user query with async processing
    """
    if not model or not tokenizer:
        return "AI model not available"

    try:
        # Split content into smaller chunks for faster processing
        chunks = chunk_content(content, chunk_size=1500)  # Adjusted chunk size
        chunk_count = len(chunks)
        print(f"\nProcessing {chunk_count} chunks of content...")

        results = []
        
        # Process chunks with increased parallelism
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            futures = []

            # Create chunks of chunks for batch processing
            batch_size = 10
            chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            
            with tqdm(total=len(chunks), desc="Processing chunks", ncols=80) as pbar:
                for batch in chunk_batches:
                    batch_futures = []
                    for chunk in batch:
                        future = loop.run_in_executor(
                            executor,
                            process_chunk_with_ai,
                            model,
                            tokenizer,
                            chunk,
                            query
                        )
                        batch_futures.append(future)

                    # Process batch with timeout
                    try:
                        batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)
                        for result in batch_results:
                            if isinstance(result, Exception):
                                logger.error(f"Error in batch processing: {str(result)}")
                                continue
                            if result and result != "No matching content found" and not result.startswith("Error"):
                                for line in result.split('\n'):
                                    line = line.strip()
                                    if line:
                                        results.append(line)
                            pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        pbar.update(len(batch))
                        continue

        # Remove duplicates while preserving order
        seen = set()
        final_results = []
        for item in results:
            item = item.strip()
            if item and item not in seen:
                seen.add(item)
                final_results.append(item)

        if not final_results:
            return "No matching content found"

        return '\n'.join(final_results)

    except Exception as e:
        logger.error(f"Error in parse_content_with_ai_async: {str(e)}")
        return f"Error in AI parsing: {str(e)}"

def save_ai_results_to_csv(ai_data, alias_name):
    """
    Save AI parsing results to CSV file with pandas
    """
    # Clean the result before saving
    result = ai_data['result']
    if isinstance(result, str):
        # Remove any SQL-like queries or command-like text
        result = re.sub(r'SELECT.*?FROM.*?WHERE.*?;?\s*', '', result, flags=re.IGNORECASE|re.MULTILINE)
        result = re.sub(r'#.*$', '', result, flags=re.MULTILINE)
        result = re.sub(r'git.*$', '', result, flags=re.MULTILINE)
        result = re.sub(r'https?://\S+', '', result)
        # Clean up empty lines and whitespace
        result = '\n'.join(line.strip() for line in result.split('\n') if line.strip())
    
    # Create DataFrame from AI data
    df = pd.DataFrame({
        'query': [ai_data['query']],
        'result': [result],
        'parsed_at': [ai_data['parsed_at']]
    })
    
    # Generate filename with alias
    filename = f"{alias_name}_ai_results.csv"
    
    # Save to CSV
    df.to_csv(filename, index=False)
    return filename

def save_scraped_data_to_csv(data, alias_name):
    """
    Save scraped data to CSV file with pandas
    """
    # Create DataFrame from scraped data
    df = pd.DataFrame({
        'title': [data['title']],
        'url': [data['url']],
        'domain': [data['domain']],
        'scraped_at': [data['scraped_at']],
        'status': [data['status']],
        'all_anchor_href': ['\n'.join(data['all_anchor_href'])],
        'all_anchors': ['\n'.join(data['all_anchors'])],
        'all_images_data': ['\n'.join(data['all_images_data'])],
        'all_images_source_data': ['\n'.join(data['all_images_source_data'])],
        'all_h1_data': ['\n'.join(data['all_h1_data'])],
        'all_h2_data': ['\n'.join(data['all_h2_data'])],
        'all_h3_data': ['\n'.join(data['all_h3_data'])],
        'all_p_data': ['\n'.join(data['all_p_data'])]
    })
    
    # Generate filename with alias
    filename = f"{alias_name}_scraped_data.csv"
    
    # Save to CSV
    df.to_csv(filename, index=False)
    return filename

# Data Export Functions
def save_to_excel(data, filename=None):
    """
    Save data to Excel file
    """
    if filename is None:
        filename = get_timestamp_filename("scraped_data", "xlsx")

    # Create a DataFrame from the data
    df = pd.DataFrame([data])

    # Save to Excel
    df.to_excel(filename, index=False)
    return filename

def save_to_csv(data, filename=None):
    """
    Save data to CSV file
    """
    if filename is None:
        filename = get_timestamp_filename("scraped_data", "csv")

    # Create a DataFrame from the data
    df = pd.DataFrame([data])

    # Save to CSV
    df.to_csv(filename, index=False)
    return filename

def save_ai_results_to_excel(ai_data, filename=None):
    """
    Save AI parsing results to Excel file
    """
    if filename is None:
        filename = get_timestamp_filename("ai_results", "xlsx")

    # Create a DataFrame from the AI data
    df = pd.DataFrame([{
        'query': ai_data['query'],
        'result': ai_data['result'],
        'parsed_at': ai_data['parsed_at']
    }])

    # Save to Excel
    df.to_excel(filename, index=False)
    return filename

# Data Visualization
def visualize_scraped_data(data):
    """
    Create visualizations of scraped data
    """
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Content Type Distribution
    content_types = {
        'Links': len(data['all_anchor_href']),
        'Images': len(data['all_images_data']),
        'H1 Headers': len(data['all_h1_data']),
        'H2 Headers': len(data['all_h2_data']),
        'H3 Headers': len(data['all_h3_data']),
        'Paragraphs': len(data['all_p_data'])
    }

    sns.barplot(x=list(content_types.values()), y=list(content_types.keys()), ax=ax1)
    ax1.set_title('Content Type Distribution')

    # Plot 2: Heading Distribution
    headings = {
        'H1': len(data['all_h1_data']),
        'H2': len(data['all_h2_data']),
        'H3': len(data['all_h3_data'])
    }

    sns.barplot(x=list(headings.keys()), y=list(headings.values()), ax=ax2)
    ax2.set_title('Heading Distribution')

    # Plot 3: Link Types
    if data['all_anchor_href']:
        link_types = pd.Series(data['all_anchor_href']).apply(lambda x: urlparse(x).netloc).value_counts().head(5)
        sns.barplot(x=link_types.values, y=link_types.index, ax=ax3)
        ax3.set_title('Top 5 Link Domains')

    # Plot 4: Image Sources
    if data['all_images_source_data']:
        image_sources = pd.Series(data['all_images_source_data']).apply(lambda x: urlparse(x).netloc).value_counts().head(5)
        sns.barplot(x=image_sources.values, y=image_sources.index, ax=ax4)
        ax4.set_title('Top 5 Image Sources')

    plt.tight_layout()
    plt.show()

# Main Program
async def main():
    try:
        # Initialize global variables
        global model, tokenizer

        # Get Colab GPU info
        get_colab_gpu_info()
        print("Note: Running with Colab optimizations")
        print("Setting up AI model...")

        # Clear memory before setup
        clear_memory()

        # Get token from environment first
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("\nTo use AI features, you need a Hugging Face token.")
            print("Get your token from: https://huggingface.co/settings/tokens")
            try:
                token = input("Enter your Hugging Face token (or press Enter to skip AI features): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSkipping AI features...")
                token = None

        # Setup AI model
        if token:
            model, tokenizer = setup_ai_model(token)
            if model and tokenizer:
                print("AI model loaded successfully!")
            else:
                print("Failed to load AI model. Running without AI features.")
        else:
            print("Running without AI features.")

        # Initialize cache for storing results
        cache = {}

        while True:
            try:
                print("""
                ================ Welcome to WebScrape =============
                ==>> press 1 for checking existing scraped websites
                ==>> press 2 for scrap a single website
                ==>> press 3 for exit
                """)

                try:
                    choice = input("==>> Please enter your choice: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting...")
                    break

                # load data
                local_json_db = load_json()
                existing_scraped_data_init(local_json_db)

                if choice == '1':
                    # Display existing scraped websites
                    scraped_websites_table = BeautifulTable()
                    scraped_websites_table.columns.header = ["Sr no.", "Alias name", "Website domain", "title", "Scraped at", "Status"]
                    scraped_websites_table.set_style(BeautifulTable.STYLE_BOX_DOUBLED)

                    for count, data in enumerate(local_json_db['scraped_data']):
                        scraped_websites_table.rows.append([
                            count + 1,
                            local_json_db['scraped_data'][data]['alias'],
                            local_json_db['scraped_data'][data]['domain'],
                            local_json_db['scraped_data'][data]['title'],
                            local_json_db['scraped_data'][data]['scraped_at'],
                            local_json_db['scraped_data'][data]['status']
                        ])

                    if not local_json_db['scraped_data']:
                        print('===> No existing data found!')
                    print(scraped_websites_table)

                elif choice == '2':
                    try:
                        url_for_scrap = input("===> Please enter url you want to scrap: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        print("\nCancelled...")
                        continue

                    # Check if URL is in cache
                    if url_for_scrap in cache:
                        print(f"Cache hit for {url_for_scrap}")
                        scraped_data_packet = cache[url_for_scrap]
                    else:
                        is_accessible = process_url_request(url_for_scrap)
                        if not is_accessible:
                            print("Error: Could not access the website. Please check the URL and try again.")
                            continue

                        scraped_data_packet = proccess_beautiful_soup_data(is_accessible)
                        print('=====> Data scraped successfully!')

                        # Cache the scraped data
                        cache[url_for_scrap] = scraped_data_packet

                    try:
                        # Get alias name and ensure it's unique
                        key_for_storing_data = input("Enter alias name for saving scraped data: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        print("\nCancelled...")
                        continue

                    original_alias = key_for_storing_data

                    # Check if alias exists and modify if necessary
                    counter = 1
                    while key_for_storing_data in local_json_db.get('scraped_data', {}):
                        key_for_storing_data = f"{original_alias}_{counter}"
                        counter += 1

                    # Update the data packet with correct information
                    scraped_data_packet.update({
                        'url': url_for_scrap,
                        'alias': key_for_storing_data,  # Use the potentially modified alias
                        'name': original_alias,  # Store original name separately
                        'scraped_at': scraped_time_is(),
                        'status': True,
                        'domain': urlparse(url_for_scrap).netloc
                    })

                    # Process with AI if available
                    if model and tokenizer:
                        print("\n=====> AI Parsing")
                        print("What information would you like to extract from this website?")
                        print("Examples:")
                        print("- 'Find all contact information'")
                        print("- 'Extract all product prices'")
                        print("- 'List all article titles'")
                        print("- 'Find social media links'")
                        print("- 'Extract all headings and their hierarchy'")
                        print("- 'Find all images and their descriptions'")
                        print("- 'Extract all links and their text'")
                        print("- 'Find all paragraphs containing specific keywords'")
                        
                        try:
                            user_query = input("Enter your query: ").strip()
                        except (KeyboardInterrupt, EOFError):
                            print("\nCancelled AI parsing...")
                            user_query = None

                        if user_query:
                            # Extract text content for AI processing
                            text_content = extract_text_content(scraped_data_packet)

                            # Parse content with AI
                            print("\nProcessing with AI...")
                            try:
                                # Process with AI
                                ai_parsed_data = await parse_content_with_ai_async(model, tokenizer, text_content, user_query)
                                
                                # Add AI parsed data to the packet
                                scraped_data_packet['ai_parsed_data'] = {
                                    'query': user_query,
                                    'result': ai_parsed_data,
                                    'parsed_at': scraped_time_is()
                                }

                                print("\n=====> AI Parsing Results:")
                                print(ai_parsed_data)

                                # Save AI results to CSV with alias
                                ai_csv_file = save_ai_results_to_csv(scraped_data_packet['ai_parsed_data'], key_for_storing_data)
                                print(f"AI results saved to CSV: {ai_csv_file}")
                                
                                # Save scraped data to CSV with alias
                                scraped_csv_file = save_scraped_data_to_csv(scraped_data_packet, key_for_storing_data)
                                print(f"Scraped data saved to CSV: {scraped_csv_file}")

                                # Save to Excel if requested
                                try:
                                    save_excel = input("Would you like to save the results to Excel as well? (y/n): ").strip().lower()
                                    if save_excel == 'y':
                                        excel_file = save_to_excel(scraped_data_packet)
                                        print(f"Data saved to Excel: {excel_file}")
                                except (KeyboardInterrupt, EOFError):
                                    print("\nSkipping Excel export...")
                                
                            except Exception as e:
                                logger.error(f"Error during AI parsing: {str(e)}")
                                print(f"Error during AI parsing: {str(e)}")
                                print("Continuing without AI results...")

                    # Save to JSON with correct alias
                    if 'scraped_data' not in local_json_db:
                        local_json_db['scraped_data'] = {}
                    local_json_db['scraped_data'][key_for_storing_data] = scraped_data_packet
                    save_scraped_data_in_json(local_json_db)

                    print('\n=====> Data saved to:')
                    print(f"- JSON: scraped_data.json (with alias: {key_for_storing_data})")
                    print(f"- CSV: {key_for_storing_data}_scraped_data.csv")
                    if 'ai_parsed_data' in scraped_data_packet:
                        print(f"- AI Results: {key_for_storing_data}_ai_results.csv")

                    # Visualize the data
                    try:
                        visualize = input("Would you like to visualize the scraped data? (y/n): ").strip().lower()
                        if visualize == 'y':
                            visualize_scraped_data(scraped_data_packet)
                    except (KeyboardInterrupt, EOFError):
                        print("\nSkipping visualization...")

                    print('=====> All data saved successfully!')

                elif choice == '3':
                    print('Thank you for using WebScrape!')
                    break

                else:
                    print("Please enter a valid choice")

            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"An error occurred: {str(e)}")
                print("Please try again.")
                continue

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        print(f"A fatal error occurred: {str(e)}")
    finally:
        # Cleanup
        if 'model' in globals():
            del model
        if 'tokenizer' in globals():
            del tokenizer

# Run the program
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting WebScrape...")
    except Exception as e:
        print(f"Error running WebScrape: {str(e)}")
    finally:
        # Ensure proper cleanup
        if 'model' in globals():
            del model
        if 'tokenizer' in globals():
            del tokenizer