import asyncio
import time

sem = asyncio.Semaphore(5)  # Limit to 5 concurrent fetches

def make_urls(n):
    return [f"https://example.com/resource/{i}" for i in range(n)]

# Writing a function that fetches one URL with semaphore and timeout handling
async def fetch_one(url, timeout=5):
    async with sem:
        try:
            async with asyncio.timeout(timeout):
                print(f"Fetching {url}...")
                await asyncio.sleep(1)  # Simulating network delay
                print(f"Finished fetching {url}.")
        except asyncio.TimeoutError:
            print(f"Timeout occurred while fetching {url}.")


# Writing a function that fetches 'n' URLs sequentially
async def fetch_urls_sequentially(n):
    start = time.perf_counter()
    urls = make_urls(n)

    for url in urls:
        await fetch_one(url)

    return time.perf_counter() - start


# Writing a function that fetches 'n' URLs concurrently
async def fetch_urls_concurrently(n):
    start = time.perf_counter()

    urls = make_urls(n)
    tasks = [fetch_one(url) for url in urls]
    await asyncio.gather(*tasks)

    return time.perf_counter() - start
