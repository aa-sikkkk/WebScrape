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

# Prompt templates for Ollama model
summarize_template = (
    "You are a helpful assistant. Summarize the following article in a concise, clear paragraph. Do not include any extra commentary.\n\nARTICLE:\n{main_text}"
)
sentiment_template = (
    "You are a sentiment analysis expert. Analyze the sentiment of the following article and respond with only one word: POSITIVE, NEGATIVE, or NEUTRAL.\n\nARTICLE:\n{main_text}"
)

# Loading JSON data
def load_json(database_json_file="scraped_data.json"):
    try:
        with open(database_json_file, "r") as read_it:
            all_data_base = json.loads(read_it.read())
            return all_data_base
    except FileNotFoundError:
        return {"scraped_data": {}}

def save_scraped_data_in_json(data, database_json_file="scraped_data.json"):
    with open(database_json_file, "w") as file_obj:
        file_obj.write(json.dumps(data, indent=4))

def scraped_time_is():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

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

def extract_main_article_text(soup):
    article = soup.find('article')
    if article:
        return article.get_text(separator='\n', strip=True)
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    return '\n'.join(paragraphs)

def safe_join(items, sep='\n'):
    return sep.join([str(x) if x is not None else '' for x in items])

def prompt_ollama(prompt):
    return llm(prompt)

def summarize_with_ollama(main_text):
    prompt = summarize_template.format(main_text=main_text)
    return prompt_ollama(prompt).strip()

def sentiment_with_ollama(main_text, chunk_size=400):
    words = main_text.split()
    sentiments = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        prompt = sentiment_template.format(main_text=chunk)
        result = prompt_ollama(prompt).strip().upper()
        # Only accept POSITIVE, NEGATIVE, NEUTRAL
        if 'POSITIVE' in result:
            sentiments.append('POSITIVE')
        elif 'NEGATIVE' in result:
            sentiments.append('NEGATIVE')
        else:
            sentiments.append('NEUTRAL')
    from collections import Counter
    if sentiments:
        most_common = Counter(sentiments).most_common(1)[0][0]
        return {
            'chunk_sentiments': sentiments,
            'overall_sentiment': most_common
        }
    else:
        return {'chunk_sentiments': [], 'overall_sentiment': 'NEUTRAL'}

async def main():
    cache = {}
    json_db = load_json()
    while True:
        print("""  ================ Welcome to this scraping program ==============
        ==>> press 1 for checking existing scraped websites
        ==>> press 2 to scrape a single website
        ==>> press 3 to exit
        """)
        try:
            choice = int(input("==>> Please enter your choice: "))
        except Exception:
            print("Invalid input. Try again.")
            continue

        if choice == 1:
            scraped = json_db.get('scraped_data', {})
            if not scraped:
                print('===> No existing data found!')
            else:
                print("Existing scraped websites:")
                for idx, alias in enumerate(scraped, 1):
                    print(f"{idx}. Alias: {alias}, Title: {scraped[alias].get('title', '')}, URL: {scraped[alias].get('url', '')}")
        elif choice == 2:
            url_for_scrap = input("===> Please enter the URL you want to scrape: ").strip()
            alias = input("Enter alias name for saving scraped data: ").strip()
            # Ensure alias is unique
            original_alias = alias
            counter = 1
            while alias in json_db.get('scraped_data', {}):
                alias = f"{original_alias}_{counter}"
                counter += 1
            if url_for_scrap in cache:
                print(f"Cache hit for {url_for_scrap}")
                soup = cache[url_for_scrap]
            else:
                soup = await process_url_request(url_for_scrap)
                cache[url_for_scrap] = soup
            scraped_data_packet = process_beautiful_soup_data(soup)
            scraped_data_packet.update({
                'url': url_for_scrap,
                'alias': alias,
                'scraped_at': scraped_time_is(),
                'status': True,
                'domain': urlparse(url_for_scrap).netloc
            })
            # Extract main article text
            print("\nExtracting main article text for advanced analysis...")
            main_text = extract_main_article_text(soup)
            if not main_text.strip():
                print("Could not extract main article text. Using all paragraphs as fallback.")
                main_text = safe_join(scraped_data_packet.get('all_p_data', []))
            print("\nChoose an advanced analysis option:")
            print("1. Summarize the main article")
            print("2. Sentiment analysis of the main article")
            print("3. Both summary and sentiment analysis")
            print("4. Skip advanced analysis")
            adv_choice = input("Enter your choice (1/2/3/4): ").strip()
            if adv_choice == '1':
                summary = summarize_with_ollama(main_text)
                print("\n=====> Summary of the main article:\n", summary)
                scraped_data_packet['summary'] = summary
            elif adv_choice == '2':
                sentiment = sentiment_with_ollama(main_text)
                print("\n=====> Sentiment analysis of the main article:\n", sentiment)
                scraped_data_packet['sentiment'] = sentiment
            elif adv_choice == '3':
                summary = summarize_with_ollama(main_text)
                sentiment = sentiment_with_ollama(main_text)
                print("\n=====> Summary of the main article:\n", summary)
                print("\n=====> Sentiment analysis of the main article:\n", sentiment)
                scraped_data_packet['summary'] = summary
                scraped_data_packet['sentiment'] = sentiment
            else:
                print("Skipping advanced analysis.")
            # Save to JSON with correct alias
            if 'scraped_data' not in json_db:
                json_db['scraped_data'] = {}
            json_db['scraped_data'][alias] = scraped_data_packet
            save_scraped_data_in_json(json_db)
            print(f'\n=====> Data saved to: scraped_data.json (with alias: {alias})')
        elif choice == 3:
            print('Thank you for using the scraper!')
            break
        else:
            print('Invalid choice. Try again.')

if __name__ == "__main__":
    asyncio.run(main())
