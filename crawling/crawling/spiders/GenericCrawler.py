import json
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import scrapy


class GenericCrawler(scrapy.Spider):
    name = "generic_crawler"
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'FEED_EXPORT_ENCODING': 'utf-8',
    }

    def __init__(self, start_url=None, allowed_domain=None, max_pages=100, max_depth=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if start_url:
            self.start_urls = [start_url]
            parsed = urlparse(start_url)
            self.allowed_domain = allowed_domain or parsed.netloc
        else:
            raise ValueError('Please provide `start_url` when launching the spider (use -a start_url=...)')

        self.max_pages = int(max_pages)
        self.max_depth = int(max_depth)
        self.pages_crawled = 0
        self.seen = set()
        self.output_path = 'output_docs.jsonl'

    def parse(self, response):
        # Basic page limiting
        if self.pages_crawled >= self.max_pages:
            return

        url = response.url
        if url in self.seen:
            return
        self.seen.add(url)

        depth = response.meta.get('depth', 0)
        if depth > self.max_depth:
            return

        # Extract text
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)

        doc_id = hashlib.sha1(url.encode('utf-8')).hexdigest()
        doc = {
            'id': doc_id,
            'url': url,
            'depth': depth,
            'text': text,
        }

        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        self.pages_crawled += 1

        # Follow same-domain links
        for link in response.css('a::attr(href)').getall():
            absolute = urljoin(url, link)
            parsed = urlparse(absolute)
            if parsed.netloc and self.allowed_domain in parsed.netloc:
                yield response.follow(absolute, callback=self.parse)
