from typing import List, Tuple, Optional

# from deepsearcher.configuration import vector_db, embedding_model, llm
from deepsearcher import configuration
from deepsearcher.vector_db.base import RetrievalResult


def query(original_query: str, max_iter: int = 3, enable_web_search: bool = False) -> Tuple[str, List[RetrievalResult], int]:
    """
    Query the knowledge base with a question and get an answer.

    This function uses the default searcher to query the knowledge base and generate
    an answer based on the retrieved information. Optionally enables web search
    for enhanced results.

    Args:
        original_query: The question or query to search for.
        max_iter: Maximum number of iterations for the search process.
        enable_web_search: Whether to enable web search using Tavily crawler.

    Returns:
        A tuple containing:
            - The generated answer as a string
            - A list of retrieval results that were used to generate the answer
            - The number of tokens consumed during the process
    """
    # Configure web search if enabled
    if enable_web_search:
        _configure_web_search()
    
    default_searcher = configuration.default_searcher
    return default_searcher.query(original_query, max_iter=max_iter)


def retrieve(
    original_query: str, max_iter: int = 3, enable_web_search: bool = False
) -> Tuple[List[RetrievalResult], List[str], int]:
    """
    Retrieve relevant information from the knowledge base without generating an answer.

    This function uses the default searcher to retrieve information from the knowledge base
    that is relevant to the query. Optionally enables web search for enhanced results.

    Args:
        original_query: The question or query to search for.
        max_iter: Maximum number of iterations for the search process.
        enable_web_search: Whether to enable web search using Tavily crawler.

    Returns:
        A tuple containing:
            - A list of retrieval results
            - An empty list (placeholder for future use)
            - The number of tokens consumed during the process
    """
    # Configure web search if enabled
    if enable_web_search:
        _configure_web_search()
    
    default_searcher = configuration.default_searcher
    retrieved_results, consume_tokens, metadata = default_searcher.retrieve(
        original_query, max_iter=max_iter
    )
    return retrieved_results, [], consume_tokens


def naive_retrieve(query: str, collection: str = None, top_k=10, enable_web_search: bool = False) -> List[RetrievalResult]:
    """
    Perform a simple retrieval from the knowledge base using the naive RAG approach.

    This function uses the naive RAG agent to retrieve information from the knowledge base
    without any advanced techniques like iterative refinement. Optionally enables web search.

    Args:
        query: The question or query to search for.
        collection: The name of the collection to search in. If None, searches in all collections.
        top_k: The maximum number of results to return.
        enable_web_search: Whether to enable web search using Tavily crawler.

    Returns:
        A list of retrieval results.
    """
    # Configure web search if enabled
    if enable_web_search:
        _configure_web_search()
    
    naive_rag = configuration.naive_rag
    all_retrieved_results, consume_tokens, _ = naive_rag.retrieve(query)
    return all_retrieved_results


def naive_rag_query(
    query: str, collection: str = None, top_k=10, enable_web_search: bool = False
) -> Tuple[str, List[RetrievalResult]]:
    """
    Query the knowledge base using the naive RAG approach and get an answer.

    This function uses the naive RAG agent to query the knowledge base and generate
    an answer based on the retrieved information, without any advanced techniques.
    Optionally enables web search for enhanced results.

    Args:
        query: The question or query to search for.
        collection: The name of the collection to search in. If None, searches in all collections.
        top_k: The maximum number of results to consider.
        enable_web_search: Whether to enable web search using Tavily crawler.

    Returns:
        A tuple containing:
            - The generated answer as a string
            - A list of retrieval results that were used to generate the answer
    """
    # Configure web search if enabled
    if enable_web_search:
        _configure_web_search()
    
    naive_rag = configuration.naive_rag
    answer, retrieved_results, consume_tokens = naive_rag.query(query)
    return answer, retrieved_results


def web_search_query(query: str, max_results: int = 5, search_depth: str = "basic") -> List[RetrievalResult]:
    """
    Perform a web search using Tavily crawler and return results.

    Args:
        query: The search query.
        max_results: Maximum number of search results to return.
        search_depth: Search depth ("basic" or "advanced").

    Returns:
        A list of retrieval results from web search.
    """
    try:
        import numpy as np
        from deepsearcher.loader.web_crawler.tavily_crawler import TavilyCrawler
        
        crawler = TavilyCrawler()
        documents = crawler.search_and_crawl(
            query=query,
            max_results=max_results,
            search_depth=search_depth
        )
        
        # Convert documents to RetrievalResult format
        results = []
        for doc in documents:
            # Create a dummy embedding (zeros) since web search doesn't provide embeddings
            dummy_embedding = np.zeros(1024)  # Standard embedding dimension
            
            result = RetrievalResult(
                embedding=dummy_embedding,
                text=doc.page_content,
                reference=doc.metadata.get("reference", ""),
                metadata=doc.metadata,
                score=doc.metadata.get("score", 0.0)
            )
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Web search failed: {e}")
        return []


def _configure_web_search():
    """
    Internal function to configure web search with Tavily crawler.
    """
    try:
        from deepsearcher.configuration import Configuration, init_config
        from deepsearcher.agent.chain_of_rag import ChainOfRAG
        from deepsearcher.agent.deep_search import DeepSearch
        from deepsearcher.agent.rag_router import RAGRouter
        from deepsearcher.agent.naive_rag import NaiveRAG
        
        # Configure Tavily crawler
        config = Configuration()
        config.set_provider_config("web_crawler", "TavilyCrawler", {})
        
        # Reinitialize configuration with web search enabled
        init_config(config)
        
        # Recreate RAG agents with web search enabled
        configuration.default_searcher = RAGRouter(
            llm=configuration.llm,
            rag_agents=[
                DeepSearch(
                    llm=configuration.llm,
                    embedding_model=configuration.embedding_model,
                    vector_db=configuration.vector_db,
                    max_iter=config.query_settings["max_iter"],
                    route_collection=True,
                    text_window_splitter=True,
                    enable_web_search=True,
                ),
                ChainOfRAG(
                    llm=configuration.llm,
                    embedding_model=configuration.embedding_model,
                    vector_db=configuration.vector_db,
                    max_iter=config.query_settings["max_iter"],
                    route_collection=True,
                    text_window_splitter=True,
                    enable_web_search=True,
                ),
            ],
        )
        
        configuration.naive_rag = NaiveRAG(
            llm=configuration.llm,
            embedding_model=configuration.embedding_model,
            vector_db=configuration.vector_db,
            top_k=10,
            route_collection=True,
            text_window_splitter=True,
            enable_web_search=True,
        )
        
    except Exception as e:
        print(f"Failed to configure web search: {e}")
