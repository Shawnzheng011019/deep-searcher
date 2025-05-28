import os
from typing import List, Optional

from langchain_core.documents import Document

from deepsearcher.loader.web_crawler.base import BaseCrawler


class TavilyCrawler(BaseCrawler):
    """
    Tavily web crawler implementation.
    
    This crawler uses the Tavily search API to search the web and extract
    content from search results, providing an intelligent search-based
    crawling approach.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Tavily crawler.

        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY environment variable.
            **kwargs: Additional keyword arguments for the crawler.
        """
        super().__init__(**kwargs)
        
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "tavily-python package is required for TavilyCrawler. "
                "Install it with: pip install tavily-python"
            )
        
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key is required. "
                "Provide it via api_key parameter or TAVILY_API_KEY environment variable."
            )
        
        self.client = TavilyClient(api_key=self.api_key)

    def crawl_url(self, url: str, **crawl_kwargs) -> List[Document]:
        """
        Extract content from a single URL using Tavily's extract API.

        Args:
            url: The URL to extract content from.
            **crawl_kwargs: Additional keyword arguments for the extraction process.

        Returns:
            A list of Document objects containing the extracted content and metadata.
        """
        try:
            include_images = crawl_kwargs.get("include_images", False)
            extract_depth = crawl_kwargs.get("extract_depth", "basic")
            
            response = self.client.extract(
                urls=url,
                include_images=include_images,
                extract_depth=extract_depth
            )
            
            documents = []
            for result in response.get("results", []):
                content = result.get("raw_content", "")
                if content:
                    metadata = {
                        "reference": result.get("url", url),
                        "source": "tavily_extract"
                    }
                    
                    # Add images to metadata if available
                    if include_images and result.get("images"):
                        metadata["images"] = result["images"]
                    
                    documents.append(Document(page_content=content, metadata=metadata))
            
            # Handle failed extractions
            for failed_result in response.get("failed_results", []):
                print(f"Warning: Failed to extract from {failed_result.get('url', url)}: {failed_result.get('error', 'Unknown error')}")
            
            return documents
            
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return []

    def search_and_crawl(self, query: str, **search_kwargs) -> List[Document]:
        """
        Search the web using Tavily and extract content from search results.

        Args:
            query: The search query.
            **search_kwargs: Additional keyword arguments for the search process.

        Returns:
            A list of Document objects containing content from search results.
        """
        try:
            max_results = search_kwargs.get("max_results", 5)
            search_depth = search_kwargs.get("search_depth", "basic")
            include_raw_content = search_kwargs.get("include_raw_content", True)
            include_images = search_kwargs.get("include_images", False)
            include_domains = search_kwargs.get("include_domains", [])
            exclude_domains = search_kwargs.get("exclude_domains", [])
            
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_raw_content=include_raw_content,
                include_images=include_images,
                include_domains=include_domains,
                exclude_domains=exclude_domains
            )
            
            documents = []
            for result in response.get("results", []):
                # Use raw_content if available, otherwise use content snippet
                content = result.get("raw_content") or result.get("content", "")
                if content:
                    metadata = {
                        "reference": result.get("url", ""),
                        "title": result.get("title", ""),
                        "score": result.get("score", 0.0),
                        "source": "tavily_search"
                    }
                    
                    # Add published date if available (for news searches)
                    if result.get("published_date"):
                        metadata["published_date"] = result["published_date"]
                    
                    documents.append(Document(page_content=content, metadata=metadata))
            
            # Add answer to documents if available
            if response.get("answer"):
                answer_metadata = {
                    "reference": "tavily_answer",
                    "source": "tavily_answer",
                    "query": query
                }
                documents.append(Document(page_content=response["answer"], metadata=answer_metadata))
            
            # Add images information if available
            if response.get("images"):
                images_content = "\n".join([
                    f"Image: {img if isinstance(img, str) else img.get('url', '')}" + 
                    (f" - {img.get('description', '')}" if isinstance(img, dict) and img.get('description') else "")
                    for img in response["images"]
                ])
                if images_content:
                    images_metadata = {
                        "reference": "tavily_images",
                        "source": "tavily_images",
                        "query": query
                    }
                    documents.append(Document(page_content=images_content, metadata=images_metadata))
            
            return documents
            
        except Exception as e:
            print(f"Error searching with query '{query}': {str(e)}")
            return []

    def get_search_context(self, query: str, **kwargs) -> str:
        """
        Get search context for RAG applications using Tavily's context API.

        Args:
            query: The search query.
            **kwargs: Additional keyword arguments.

        Returns:
            A context string suitable for RAG applications.
        """
        try:
            return self.client.get_search_context(query=query, **kwargs)
        except Exception as e:
            print(f"Error getting search context for query '{query}': {str(e)}")
            return ""

    def qna_search(self, query: str, **kwargs) -> str:
        """
        Get a quick answer to a question using Tavily's QnA API.

        Args:
            query: The question to answer.
            **kwargs: Additional keyword arguments.

        Returns:
            An answer string.
        """
        try:
            return self.client.qna_search(query=query, **kwargs)
        except Exception as e:
            print(f"Error getting QnA answer for query '{query}': {str(e)}")
            return "" 