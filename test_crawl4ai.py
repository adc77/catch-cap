# import asyncio
# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
# from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
# from crawl4ai.deep_crawling.filters import (
#     FilterChain,
#     DomainFilter,
#     URLPatternFilter,
#     ContentTypeFilter
# )
# from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

# async def run_advanced_crawler():
#     # Create a sophisticated filter chain
#     filter_chain = FilterChain([
#         # Domain boundaries
#         DomainFilter(
#             allowed_domains=[
#                 "www.nishithdesai.com",
#                 "www.cyrilshroff.com", 
#                 "www.nlsir.com",
#                 "www.azbpartners.com",
#                 "www.amsshardul.com",
#                 "www.mondaq.com",
#                 "clpr.org.in",
#                 "vidhilegalpolicy.in",
#                 "trustbridge.in",
#                 "repository.nls.ac.in",
#                 "www.pudr.org",
#                 "www.ikigailaw.com",
#                 "nujslawreview.org",
#                 "nlujlawreview.in",
#                 "jgu.edu.in",
#                 "nliulawreview.nliu.ac.in",
#                 "nslr.in",
#                 "elplaw.in",
#                 "dsklegal.com",
#                 "luthra.com",
#                 "www.jsalaw.com",
#                 "law.asia",
#                 "liiofindia.org",
#                 "hollis.harvard.edu",
#                 "blogs.law.ox.ac.uk",
#                 "peacepalacelibrary.nl",
#                 "legal.un.org"
#             ],
#             blocked_domains=["old.docs.example.com"]
#         ),

#         # URL patterns to include
#         URLPatternFilter(patterns=["*guide*", "*tutorial*", "*blog*"]),

#         # Content type filtering
#         ContentTypeFilter(allowed_types=["text/html"])
#     ])

#     # Create a relevance scorer
#     keyword_scorer = KeywordRelevanceScorer(
#         keywords=["crawl", "example", "async", "configuration"],
#         weight=0.7
#     )

#     # Set up the configuration
#     config = CrawlerRunConfig(
#         deep_crawl_strategy=BestFirstCrawlingStrategy(
#             max_depth=2,
#             include_external=False,
#             filter_chain=filter_chain,
#             url_scorer=keyword_scorer
#         ),
#         scraping_strategy=LXMLWebScrapingStrategy(),
#         stream=True,
#         verbose=True
#     )

#     # Execute the crawl
#     results = []
#     async with AsyncWebCrawler() as crawler:
#         async for result in await crawler.arun("https://docs.crawl4ai.com/", config=config):
#             results.append(result)
#             score = result.metadata.get("score", 0)
#             depth = result.metadata.get("depth", 0)
#             print(f"Depth: {depth} | Score: {score:.2f} | {result.url}")

#     # Analyze the results
#     crawled_pages = len(results)
#     average_score = sum(r.metadata.get('score', 0) for r in results) / crawled_pages
#     depth_counts = {}
#     for result in results:
#         depth = result.metadata.get("depth", 0)
#         depth_counts[depth] = depth_counts.get(depth, 0) + 1

#     # Save results to markdown file
#     with open("test_crawl4ai.md", "w") as f:
#         f.write(f"# Crawl Results\n\n")
#         f.write(f"Crawled {crawled_pages} high-value pages\n")
#         f.write(f"Average score: {average_score:.2f}\n\n")
#         f.write("Pages crawled by depth:\n")
#         for depth, count in sorted(depth_counts.items()):
#             f.write(f"  Depth {depth}: {count} pages\n")

# if __name__ == "__main__":
#     asyncio.run(run_advanced_crawler())



# import asyncio
# from crawl4ai import AsyncWebCrawler

# async def main():
#     async with AsyncWebCrawler() as crawler:
#         result = await crawler.arun("https://qdrant.tech/documentation/concepts/search/")
#         print(result.markdown[:900])  # Print first 300 chars

# if __name__ == "__main__":
#     asyncio.run(main())



import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# async def main():
#     browser_conf = BrowserConfig(headless=True)  # or False to see the browser
#     run_conf = CrawlerRunConfig(
#         cache_mode=CacheMode.BYPASS
#     )

#     async with AsyncWebCrawler(config=browser_conf) as crawler:
#         result = await crawler.arun(
#             url="https://qdrant.tech/documentation/concepts/search/",
#             config=run_conf
#         )
#         print(result.markdown)

# if __name__ == "__main__":
#     asyncio.run(main())



# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# from crawl4ai.content_filter_strategy import PruningContentFilter
# from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# async def main():
#     md_generator = DefaultMarkdownGenerator(
#         content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
#     )
#     config = CrawlerRunConfig(
#         cache_mode=CacheMode.BYPASS,
#         markdown_generator=md_generator
#     )

#     async with AsyncWebCrawler() as crawler:
#         result = await crawler.arun("https://news.ycombinator.com", config=config)
#         print("Raw Markdown length:", len(result.markdown.raw_markdown))
#         print("Fit Markdown length:", len(result.markdown.fit_markdown))
#         print(result.markdown.fit_markdown[:1000])

# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
from crawl4ai import AsyncWebCrawler, AdaptiveCrawler

async def adaptive_example():
    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler)

        # Start adaptive crawling
        result = await adaptive.digest(
            start_url="https://www.nishithdesai.com/",
            query="In what scenarios is the termination of fixed-term contract workers is classified as retrenchment"
        )

        # View results
        adaptive.print_stats()
        print(f"Crawled {len(result.crawled_urls)} pages")
        print(f"Achieved {adaptive.confidence:.0%} confidence")
        
        with open("test.md", "w", encoding="utf-8") as file:
            for page in adaptive.get_relevant_content(top_k=3):
                file.write(page["url"] + "\n")
                file.write("--------------------------------\n")
                file.write("--------------CONTENT------------------\n")
                file.write(page["content"] + "\n")
                file.write("-------------END----------------\n")

if __name__ == "__main__":
    asyncio.run(adaptive_example())

"""
create a list of source/start urls
for a given query -> use a small model to select the best start_url (model will be used from openai_client.py)
then convert the complete query to searchable query (keywords++)
then use the adaptive crawler to crawl the best start_url
and get the most relevant urls to the given query!

"""

# import asyncio
# import time
# from crawl4ai.async_webcrawler import AsyncWebCrawler, CacheMode
# from crawl4ai.async_configs import CrawlerRunConfig
# from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, RateLimiter

# VERBOSE = False

# async def crawl_sequential(urls):
#     config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=VERBOSE)
#     results = []
#     start_time = time.perf_counter()
#     async with AsyncWebCrawler() as crawler:
#         for url in urls:
#             result_container = await crawler.arun(url=url, config=config)
#             results.append(result_container[0])
#     total_time = time.perf_counter() - start_time
#     return total_time, results

# async def crawl_parallel_dispatcher(urls):
#     config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=VERBOSE)
#     # Dispatcher with rate limiter enabled (default behavior)
#     dispatcher = MemoryAdaptiveDispatcher(
#         rate_limiter=RateLimiter(base_delay=(1.0, 3.0), max_delay=60.0, max_retries=3),
#         max_session_permit=50,
#     )
#     start_time = time.perf_counter()
#     async with AsyncWebCrawler() as crawler:
#         result_container = await crawler.arun_many(urls=urls, config=config, dispatcher=dispatcher)
#         results = []
#         if isinstance(result_container, list):
#             results = result_container
#         else:
#             async for res in result_container:
#                 results.append(res)
#     total_time = time.perf_counter() - start_time
#     return total_time, results

# async def crawl_parallel_no_rate_limit(urls):
#     config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=VERBOSE)
#     # Dispatcher with no rate limiter and a high session permit to avoid queuing
#     dispatcher = MemoryAdaptiveDispatcher(
#         rate_limiter=None,
#         max_session_permit=len(urls)  # allow all URLs concurrently
#     )
#     start_time = time.perf_counter()
#     async with AsyncWebCrawler() as crawler:
#         result_container = await crawler.arun_many(urls=urls, config=config, dispatcher=dispatcher)
#         results = []
#         if isinstance(result_container, list):
#             results = result_container
#         else:
#             async for res in result_container:
#                 results.append(res)
#     total_time = time.perf_counter() - start_time
#     return total_time, results

# async def main():
#     urls = ["https://example.com"] * 100
#     print(f"Crawling {len(urls)} URLs sequentially...")
#     seq_time, seq_results = await crawl_sequential(urls)
#     print(f"Sequential crawling took: {seq_time:.2f} seconds\n")
    
#     print(f"Crawling {len(urls)} URLs in parallel using arun_many with dispatcher (with rate limit)...")
#     disp_time, disp_results = await crawl_parallel_dispatcher(urls)
#     print(f"Parallel (dispatcher with rate limiter) took: {disp_time:.2f} seconds\n")
       
#     print(f"Crawling {len(urls)} URLs in parallel using dispatcher with no rate limiter...")
#     no_rl_time, no_rl_results = await crawl_parallel_no_rate_limit(urls)
#     print(f"Parallel (dispatcher without rate limiter) took: {no_rl_time:.2f} seconds\n")
    
#     print("Crawl4ai - Crawling Comparison")
#     print("--------------------------------------------------------")
#     print(f"Sequential crawling took: {seq_time:.2f} seconds")
#     print(f"Parallel (dispatcher with rate limiter) took: {disp_time:.2f} seconds")
#     print(f"Parallel (dispatcher without rate limiter) took: {no_rl_time:.2f} seconds")
    
# if __name__ == "__main__":
#     asyncio.run(main())
