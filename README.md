# WebScrape
WebScrape is a simple yet powerful Python-based web scraping tool that allows users to extract and store website data, including titles, anchor tags, images, headings, and paragraphs. The scraped data is stored in a JSON file and can be viewed in a tabular format using the `BeautifulTable` library.

## Features

- Scrape web pages to extract:
  - Title
  - Anchor tags (`<a>`)
  - Images (`<img>`)
  - Headings (`<h1>`, `<h2>`, `<h3>`)
  - Paragraphs (`<p>`)
- Stores scraped data in a JSON file for future use.
- Displays existing scraped websites in a user-friendly table format.
- Allows alias names for websites to manage and store scraped data.
- Handles multiple websites and retains a history of scrapes.

## Google Collab
You can find the google notebook from: [link](https://colab.research.google.com/drive/1t03WODhStp3oYeFthi4r9gZuNXCR31lE?usp=sharing)

## Requirements

- Python 
- `requests` - For making HTTP requests.
- `beautifulsoup4` - For parsing the HTML.
- `lxml` - A fast XML and HTML parser.
- `beautifultable` - For displaying scraped data in a table.

Install the dependencies using the following command:
```bash
git clone https://github.com/aa-sikkkk/WebScrape.git
cd WebScrape
```

```bash
pip install -r requirements.txt
```
```bash 
python scrap.py
```

##
Data Storage

![1_poXvWu--59Gd64sRktAQGQ](https://github.com/user-attachments/assets/15011f46-de61-487c-85b2-c433c433b9a5)


```
{
    "scraped_data": {
        "alias_name": {
            "url": "http://example.com",
            "title": "Example Website",
            "all_anchor_href": [...],
            "all_anchors": [...],
            "all_images_data": [...],
            "all_images_source_data": [...],
            "all_h1_data": [...],
            "all_h2_data": [...],
            "all_h3_data": [...],
            "all_p_data": [...],
            "scraped_at": "dd/mm/yyyy hh:mm:ss",
            "status": true,
            "domain": "example.com"
        }
    }
}

```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing
Feel free to fork the project and submit pull requests! If you encounter any issues, you can open an issue on the repository.

## Acknowledgements
BeautifulSoup - For HTML parsing.
BeautifulTable - For displaying data in a table format.
