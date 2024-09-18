from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from datetime import datetime
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from .models import ScrapedData
import json


def home(request):
    return HttpResponse("Welcome to the WebScrape app!")


def download_json(request, alias):
    # Use filter() to handle cases with multiple objects and select the first one
    data = ScrapedData.objects.filter(alias=alias).first()  # Retrieves the first entry if multiple exist

    if data:
        # Convert the data to a dictionary and handle potential JSON fields
        data_dict = {
            'alias': data.alias,
            'url': data.url,
            'title': data.title,
            'scraped_at': data.scraped_at.strftime('%d/%m/%Y %H:%M:%S') if data.scraped_at else 'N/A',
            'status': data.status,
            'domain': data.domain,
            'all_anchor_href': json.loads(data.all_anchor_href) if data.all_anchor_href else [],
            'all_anchors': json.loads(data.all_anchors) if data.all_anchors else [],
            'all_images_data': json.loads(data.all_images_data) if data.all_images_data else [],
            'all_images_source_data': json.loads(data.all_images_source_data) if data.all_images_source_data else [],
            'all_h1_data': json.loads(data.all_h1_data) if data.all_h1_data else [],
            'all_h2_data': json.loads(data.all_h2_data) if data.all_h2_data else [],
            'all_h3_data': json.loads(data.all_h3_data) if data.all_h3_data else [],
            'all_p_data': json.loads(data.all_p_data) if data.all_p_data else []
        }
        # Convert the dictionary to a JSON string
        json_data = json.dumps(data_dict, indent=4)

        # Create an HttpResponse with the JSON data
        response = HttpResponse(json_data, content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="{alias}.json"'
        return response
    else:
        return HttpResponse("Data not found", status=404)
    
def scrape_website(request):
    if request.method == "POST":
        url = request.POST.get('url')
        alias = request.POST.get('alias')

        if not url or not alias:
            return JsonResponse({'error': 'URL and alias are required.'}, status=400)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for HTTP status codes 4xx/5xx
            soup = BeautifulSoup(response.text, 'html.parser')  # Using 'html.parser'

            title = soup.find('title').text if soup.find('title') else 'No title found'
            scraped_at = datetime.now()

            all_anchor_href = [a.get('href') for a in soup.find_all('a', href=True)]
            all_anchors = [str(a) for a in soup.find_all('a')]
            all_images_data = [str(img) for img in soup.find_all('img')]
            all_images_source_data = [img.get('src') for img in soup.find_all('img')]
            all_h1_data = [h1.text for h1 in soup.find_all('h1')]
            all_h2_data = [h2.text for h2 in soup.find_all('h2')]
            all_h3_data = [h3.text for h3 in soup.find_all('h3')]
            all_p_data = [p.text for p in soup.find_all('p')]

            # Save data to the database
            scraped_data = ScrapedData(
                alias=alias,
                url=url,
                title=title,
                scraped_at=scraped_at,
                status=True,
                domain=urlparse(url).netloc,
                all_anchor_href=all_anchor_href,
                all_anchors=all_anchors,
                all_images_data=all_images_data,
                all_images_source_data=all_images_source_data,
                all_h1_data=all_h1_data,
                all_h2_data=all_h2_data,
                all_h3_data=all_h3_data,
                all_p_data=all_p_data
            )
            scraped_data.save()

            # Redirect to the list page after saving
            return redirect('list_scraped_websites')
        except requests.RequestException as e:
            return JsonResponse({'error': str(e)}, status=500)
    return render(request, 'scrape.html')


def list_scraped_websites(request):
    scraped_data = ScrapedData.objects.all()
    return render(request, 'list.html', {'scraped_data': scraped_data})
