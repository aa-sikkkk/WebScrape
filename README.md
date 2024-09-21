<p align="center">
   <img  src="https://github.com/user-attachments/assets/da6e866b-fab6-435c-b860-aed5c13b8984"/>
   </br>
   <a href="https://github.com/aa-sikkkk/WebScrape/pulls">
       <img src="https://img.shields.io/github/issues-pr/aa-sikkkk/WebScrape" alt="Pull Requests"/>
   </a>
   <a href="https://github.com/aa-sikkkk/WebScrape/stargazers">
       <img src="https://img.shields.io/github/stars/aa-sikkkk/WebScrape" alt="Stars"/>
   </a>
   <a href="https://github.com/aa-sikkkk/WebScrape/releases">
       <img src="https://img.shields.io/github/downloads/aa-sikkkk/WebScrape/total" alt="Downloads"/>
   </a>
   <a href="https://github.com/aa-sikkkk/WebScrape/issues">
       <img src="https://img.shields.io/github/issues/aa-sikkkk/WebScrape" alt="Issues"/>
   </a>
   <a href="https://github.com/aa-sikkkk/WebScrape/blob/main/LICENSE">
       <img src="https://img.shields.io/github/license/aa-sikkkk/WebScrape" alt="License"/>
   </a>
</p>



WebScrape is a simple yet powerful Python-based web scraping tool that allows users to extract and store website data, including titles, anchor tags, images, headings, and paragraphs. The scraped data is stored in a JSON file and can be viewed in a tabular format using the `BeautifulTable` library


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
- Web Version as well as CLI(With Ollama Integration) Version is available.

## [📔 Google Collab](https://colab.research.google.com/drive/1t03WODhStp3oYeFthi4r9gZuNXCR31lE?usp=sharing)
You can use WebScrape on [Google Colab](https://colab.research.google.com/) **for free**. The project is using Llama model using hugging face for data parsing. if you don't have a powerful GPU of your own. You can borrow a powerful GPU (Tesla K80, T4, P4, or P100) on Google's server for free for a maximum of 12 hours per session. **Please use the free resource fairly** and do not create sessions back-to-back and run upscaling 24/7. This might result in you getting banned. You can get [Colab Pro/Pro+](https://colab.research.google.com/signup/pricing) if you'd like to use better GPUs and get longer runtimes. Usage instructions are embedded in the [Colab Notebook](https://colab.research.google.com/drive/1t03WODhStp3oYeFthi4r9gZuNXCR31lE?usp=sharing). Check out the [wiki page.](https://github.com/aa-sikkkk/WebScrape/wiki)

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

## Web Version of the Project.
The Project is powered by Django for web version.

![Screenshot 2024-09-18 120647](https://github.com/user-attachments/assets/389721fb-4a19-4c0c-9c90-e0dbab49c959)
![Screenshot 2024-09-18 120628](https://github.com/user-attachments/assets/06ec5a10-7210-4e00-b9c3-5c8a749048b5)

## License
This project is licensed under the MIT [License](LICENSE). See the LICENSE file for more details.

## Contributing
Feel free to fork the project and submit pull requests! If you encounter any issues, you can open an issue on the repository.

## Acknowledgements
BeautifulSoup - For HTML parsing.
BeautifulTable - For displaying data in a table format.
