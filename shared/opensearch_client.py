"""
Shared OpenSearch client initialization with configurable connection pool size.
Used by all retrievers and search modules to avoid code duplication.
"""

import os

# Configuration
OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_MAXSIZE     = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))  # Connection pool size
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")

_os_client = None


def get_opensearch_client():
    """
    Get or create a singleton OpenSearch client with proper connection pool sizing.
    Sets urllib3 HTTPConnectionPool.maxsize to OPENSEARCH_MAXSIZE before creating client.
    """
    global _os_client
    if _os_client is None:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        import urllib3
        
        # Disable urllib3 warnings
        urllib3.disable_warnings()
        
        # Set urllib3 connection pool size BEFORE creating OpenSearch client
        # This is the actual connection pool that was showing "pool size: 10"
        from urllib3.util.connection import HTTPConnectionPool, HTTPSConnectionPool
        HTTPConnectionPool.maxsize = OPENSEARCH_MAXSIZE
        HTTPSConnectionPool.maxsize = OPENSEARCH_MAXSIZE
        
        frozen  = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(
            frozen.access_key, frozen.secret_key, AWS_REGION, "es",
            session_token=frozen.token
        )
        
        _os_client = OpenSearch(
            hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=2,
            retry_on_timeout=True,
            maxsize=OPENSEARCH_MAXSIZE,  # OpenSearch client's connection pool size
        )
    
    return _os_client


def reset_client():
    """Reset the client singleton (for testing)."""
    global _os_client
    _os_client = None
