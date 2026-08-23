import asyncio
import time
import httpx

sem = asyncio.Semaphore(5)  # Limit to 5 concurrent fetches

BASE_URL = "https://jsonplaceholder.typicode.com/todos"


def make_urls(n):
    # jsonplaceholder only has 200 valid todo ids, so cycle through them
    # instead of generating out-of-range ids for larger n.
    return [f"{BASE_URL}/{(i % 200) + 1}" for i in range(n)]


async def fetch_one(client, url, timeout=5):
    async with sem:
        try:
            async with asyncio.timeout(timeout):
                print(f"Fetching {url}...")
                response = await client.get(url)
                response.raise_for_status()
                print(f"Finished fetching {url} -> status {response.status_code}")
                return response.json()
        except asyncio.TimeoutError:
            print(f"Timeout occurred while fetching {url}.")
            return None
        except httpx.HTTPError as e:
            print(f"HTTP error while fetching {url}: {e}")
            return None


# We shouldn't use a synchronus libraby like requests in an async function, 
# so we will use httpx for async HTTP requests instead of requests.

# But there will be casese when we need to use synchronous libraries in async functions, 
# so we will use asyncio.to_thread to run synchronous code in a separate thread.
async def fetch_urls_sequentially(n):
    urls = make_urls(n)
    start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        results = []
        for url in urls:
            result = await fetch_one(client, url)
            results.append(result)

    return time.perf_counter() - start, results


async def fetch_urls_concurrently(n):
    urls = make_urls(n)
    start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [fetch_one(client, url) for url in urls]
        results = await asyncio.gather(*tasks)

    return time.perf_counter() - start, results


async def run_benchmark():
    for n in (1, 10, 50):
        seq_time, _ = await fetch_urls_sequentially(n)
        conc_time, _ = await fetch_urls_concurrently(n)
        speedup = seq_time / conc_time if conc_time else float("inf")

        print(f"\nN={n}")
        print(f"  sequential : {seq_time:.2f}s")
        print(f"  concurrent : {conc_time:.2f}s")
        print(f"  speedup    : {speedup:.1f}x")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
