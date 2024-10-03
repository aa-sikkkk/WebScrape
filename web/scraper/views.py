from django.shortcuts import render, redirect
from django.http import JsonResponse
from datetime import datetime
from urllib.parse import urlparse
from .models import ScrapedData
from .scrape import scrape_website, extract_body_content, clean_body_content, split_dom_content
from .parse import parse_with_ollama
import json

def scrape_website_view(request):
    if request.method == "POST":
        url = request.POST.get('url')
        alias = request.POST.get('alias')

        if not url or not alias:
            return JsonResponse({'error': 'URL and alias are required.'}, status=400)

        # Scrape the website using Selenium
        try:
            dom_content = scrape_website(url)
            body_content = extract_body_content(dom_content)
            cleaned_content = clean_body_content(body_content)

            # Store the scraped data in the database
            scraped_data = ScrapedData(
                alias=alias,
                url=url,
                title='Scraped Title',  # Consider extracting the actual title
                scraped_at=datetime.now(),
                status=True,
                domain=urlparse(url).netloc,
                all_anchor_href=json.dumps([]),  # Update with actual data if needed
                all_anchors=json.dumps([]),       # Update with actual data if needed
                all_images_data=json.dumps([]),   # Update with actual data if needed
                all_images_source_data=json.dumps([]),  # Update with actual data if needed
                all_h1_data=json.dumps([]),       # Update with actual data if needed
                all_h2_data=json.dumps([]),       # Update with actual data if needed
                all_h3_data=json.dumps([]),       # Update with actual data if needed
                all_p_data=json.dumps([])         # Update with actual data if needed
            )
            scraped_data.save()

            # Redirect to the list page after saving
            return redirect('list_scraped_websites')
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return render(request, 'scrape.html')

def parse_content_view(request):
    if request.method == "POST":
        parse_description = request.POST.get('parse_description')

        # Retrieve the cleaned content from the session
        dom_content = request.session.get('dom_content')

        if not dom_content or not parse_description:
            return JsonResponse({'error': 'DOM content and parse description are required.'}, status=400)

        dom_chunks = split_dom_content(dom_content)
        parsed_result = parse_with_ollama(dom_chunks, parse_description)

        return JsonResponse({'parsed_result': parsed_result})

    return render(request, 'scrape.html')  # Render the form page if GET request
