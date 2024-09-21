import json
import asyncio
import aiohttp
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain.llms import Ollama  # for OLLAMA

# Initialize the Ollama model locally
llm = Ollama(model="llama-2-7b")

# Prompt template for Ollama model
template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
)

# Loading JSON data
def load_json(database_json_file="scraped_data.json"):
    try:
        with open(database_json_file, "r") as read_it:
            all_data_base = json.loads(read_it.read())
            return all_data_base
    except FileNotFoundError:
        return {"scraped_data": {}}

# Saving data to JSON
def save_scraped_data_in_json(data, database_json_file="scraped_data.json"):
    with open(database_json_file, "w") as file_obj:
        file_obj.write(json.dumps(data, indent=4))

# Getting current time in a specific format
def scraped_time_is():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

# Cleaning the HTML content
def clean_html_content(dom_content):
    soup = BeautifulSoup(dom_content, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    unwanted_tags = ['header', 'footer', 'nav', 'aside', 'form']
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()
    cleaned_content = soup.get_text(separator=" ").strip()
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
    return cleaned_content

# Spliting the content into chunks (reduce chunk size to speed up)
def chunk_content(content, chunk_size=512):
    content_words = content.split()
    return [' '.join(content_words[i:i + chunk_size]) for i in range(0, len(content_words), chunk_size)]

async def fetch_html(session, url):
    async with session.get(url) as response:
        return await response.text()

async def process_url_request(website_url):
    async with aiohttp.ClientSession() as session:
        html_content = await fetch_html(session, website_url)
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup

# Processing soup data
def process_beautiful_soup_data(soup):
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

# Batch processing of multiple chunks
async def batch_process_model(chunks, parse_description):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            loop.run_in_executor(executor, prompt_model_for_parsing, chunk, parse_description)
            for chunk in chunks
        ]
        results = []
        # For progress tracking
        for future in tqdm_asyncio.as_completed(futures, desc="Processing Chunks", total=len(futures)):
            results.append(await future)
    return results

# Prompt the Ollama model via LangChain for parsing the content
def prompt_model_for_parsing(dom_content, parse_description):
    prompt = template.format(dom_content=dom_content, parse_description=parse_description)
    result = llm(prompt)  # interracting with the Ollama model
    return result

# Function to scrape and parse the website
async def scrape_and_parse(url, parse_description, cache):
    if url in cache:
        print(f"Cache hit for {url}")
        return cache[url]

    soup = await process_url_request(url)
    scraped_data_packet = process_beautiful_soup_data(soup)

    dom_content = str(soup)
    cleaned_content = clean_html_content(dom_content)
    content_chunks = chunk_content(cleaned_content, chunk_size=512)

    print("Parsing content, please wait...")
    batched_results = await batch_process_model(content_chunks, parse_description)

    # Caching the Result 
    cache[url] = batched_results
    return batched_results

# Main program logic for scraping and parsing
async def main():
    cache = {}
    while True:
        print("""  ================ Welcome to this scraping program ==============
        ==>> press 1 for checking existing scraped websites
        ==>> press 2 to scrape a single website
        ==>> press 3 to exit
        """)

        choice = int(input("==>> Please enter your choice: "))

        if choice == 2:
            parse_description = input("Enter the specific data you want to extract (e.g., all headers, links, etc.): ")
            url_for_scrap = input("===> Please enter the URL you want to scrape: ")
            parsed_output = await scrape_and_parse(url_for_scrap, parse_description, cache)
            # Saving parsed data to a file
            timestamp = scraped_time_is().replace("/", "_").replace(" ", "_").replace(":", "_")
            json_filename = f"parsed_data_{timestamp}.json"
            with open(json_filename, "w") as f:
                json.dump(parsed_output, f, indent=4)
            print(f'Parsed data saved to {json_filename} and is ready for download.')
        elif choice == 3:
            print('Thank you for using the scraper!')
            break

# Run the main loop
if __name__ == "__main__":
    asyncio.run(main())
