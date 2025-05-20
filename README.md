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

# WebScrape

WebScrape is a powerful Python-based web scraping tool that combines traditional web scraping with AI-powered content parsing. It allows users to extract, analyze, and visualize website data with advanced features for data processing and storage.

## 🌟 Features

### Core Scraping Features
- Extract comprehensive website data:
  - Title and metadata
  - Anchor tags and links
  - Images and their sources
  - Headings (H1, H2, H3)
  - Paragraphs and text content
  - Custom content based on user queries

### AI Integration
- AI-powered content parsing using:
  - Hugging Face models (OPT-1.3B/350M)
  - Customizable queries for targeted extraction
  - Intelligent content analysis and structuring
  - Context-aware information extraction

### Data Management
- Multiple storage formats:
  - JSON for structured data storage
  - CSV for spreadsheet compatibility
  - Excel for advanced data analysis
- Data visualization capabilities
- Caching system for improved performance
- Unique alias system for data organization

### User Interface
- Command-line interface (CLI)
- Web interface powered by Django
- Interactive data visualization
- Progress tracking and feedback

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Git
- (Optional) Hugging Face token for AI features

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aa-sikkkk/WebScrape.git
cd WebScrape
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the scraper:
```bash
python web_scraper_notebook.py
```

### Using Google Colab
You can use WebScrape on [Google Colab](https://colab.research.google.com/drive/1t03WODhStp3oYeFthi4r9gZuNXCR31lE?usp=sharing) for free. The project uses Hugging Face models for data parsing.

**Note:** Please use the free resources responsibly:
- Maximum 12 hours per session
- Avoid creating back-to-back sessions
- Consider [Colab Pro/Pro+](https://colab.research.google.com/signup/pricing) for better GPUs and longer runtimes

## 📊 Data Structure

```json
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
            "domain": "example.com",
            "ai_parsed_data": {
                "query": "user query",
                "result": "parsed content",
                "parsed_at": "timestamp"
            }
        }
    }
}
```

## 🌐 Web Interface

The project includes a Django-powered web interface for easier interaction:

![Web Interface Screenshot 1](https://github.com/user-attachments/assets/389721fb-4a19-4c0c-9c90-e0dbab49c959)
![Web Interface Screenshot 2](https://github.com/user-attachments/assets/06ec5a10-7210-4e00-b9c3-5c8a749048b5)

## 🤝 Contributing

We welcome contributions! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Open issues for bugs or feature requests

## 📝 License

This project is licensed under the MIT [License](LICENSE). See the LICENSE file for details.

## 🙏 Acknowledgements

- BeautifulSoup - HTML parsing
- BeautifulTable - Data display
- Hugging Face - AI models
- Django - Web framework
- All contributors and users of the project
