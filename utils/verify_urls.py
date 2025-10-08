import aiohttp
import asyncio
import logging
from download_models import MODEL_URLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_url(session: aiohttp.ClientSession, url: str) -> bool:
    """Verify if URL is accessible"""
    try:
        async with session.get(url, allow_redirects=True) as response:
            if response.status == 404:
                logger.error(f"❌ {url} (Status: 404)")
                return False
            elif response.status == 200 or response.status == 302:
                logger.info(f"✅ {url}")
                return True
            else:
                logger.error(f"❌ {url} (Status: {response.status})")
                return False
    except Exception as e:
        logger.error(f"❌ {url} (Error: {str(e)})")
        return False

async def verify_all_urls():
    """Verify all URLs in MODEL_URLS"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        tasks = []
        for model_name, urls in MODEL_URLS.items():
            logger.info(f"\nChecking {model_name}:")
            for file_type, url in urls.items():
                tasks.append(verify_url(session, url))
        
        results = await asyncio.gather(*tasks)
        return all(results)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(verify_all_urls())
    if success:
        print("\nAll URLs are valid! ✅")
    else:
        print("\nSome URLs failed verification! ❌")
    loop.close() 