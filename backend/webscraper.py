import requests
from bs4 import BeautifulSoup
import openai

openai.api_key = 'banana'

def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None

def parse_website_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text(strip=True) for p in paragraphs])
    return text_content[:4000]

def summarize_content(content):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Harvard Data Analytics Group reaches out to companies and does data analysis and consulting for various clients. You are an assistant that, given text scraped off a company website, will summarize it so Harvard Data Analytics Group can reach out to said company about partnering with them."},
                {"role": "user", "content": f"Summarize this content to describe what the company does:\n\n{content}"}
            ]
        )
        summary = response['choices'][0]['message']['content']
        return summary
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def run_website_analytics(url):
    html = fetch_website_content(url)
    if html:
        content = parse_website_content(html)
        print(content)
        if content:
            summary = summarize_content(content)
            if summary:
                return summary
            else:
                print("Failed to generate a summary.")
        else:
            print("No relevant content found on the website.")
    else:
        print("Failed to fetch the website.")
