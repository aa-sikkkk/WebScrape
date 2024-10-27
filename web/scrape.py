from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import time

load_dotenv()

CHROME_DRIVER_PATH = os.getenv("CHROME_DRIVER_PATH")


def scrape_website(website):
    print("Starting Chrome WebDriver...")
    options = Options()
    options.add_argument("--headless")  # Use headless mode if you don't need UI
    service = Service(CHROME_DRIVER_PATH)

    with webdriver.Chrome(service=service, options=options) as driver:
        driver.get(website)
        print("Waiting for CAPTCHA to solve...")

        # Simulate waiting for a CAPTCHA to be solved
        time.sleep(10)  # Adjust as necessary for manual solving

        print("Navigated! Scraping page content...")
        html = driver.page_source
        return html


def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get text or further process the content
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content


def split_dom_content(dom_content, max_length=6000):
    return [
        dom_content[i: i + max_length] for i in range(0, len(dom_content), max_length)
    ]