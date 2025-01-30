import importlib
import os
import time
import uuid
from typing import List

import numpy as np
import pinecone  # type: ignore
import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests
from pinecone import ServerlessSpec
from pytest_mock import MockerFixture  # type: ignore[import-not-found]

import langchain_pinecone

INDEX_NAME = "langchain-test-index"  # name of the index
NAMESPACE_NAME = "langchain-test-namespace"  # name of the namespace
DIMENSION = 1536  # dimension of the embeddings

DEFAULT_SLEEP = 20

@pytest.fixture(scope="class", params=[False, True], ids=["http", "grpc"])
def get_langchain_pinecone(request):
    if not request.param:
        with pytest.MonkeyPatch.context() as monkey_patch:
            monkey_patch.delattr("pinecone.grpc", raising=False)
            importlib.reload(langchain_pinecone)
    return langchain_pinecone

class TestPinecone(VectorStoreIntegrationTests):
    index: "pinecone.Index"
    pc: "pinecone.Pinecone"

    @classmethod
    def setup_class(self) -> None:
        import pinecone

        client = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_list = client.list_indexes()
        if INDEX_NAME in [
            i["name"] for i in index_list
        ]:  # change to list comprehension
            client.delete_index(INDEX_NAME)
            time.sleep(DEFAULT_SLEEP)  # prevent race with subsequent creation
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )

        self.index = client.Index(INDEX_NAME)
        self.pc = client

    @classmethod
    def teardown_class(self) -> None:
        self.pc.delete_index()

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the vectors in the index
        print("called")  # noqa: T201
        index_stats = self.index.describe_index_stats()
        if index_stats["total_vector_count"] > 0:
            try:
                self.index.delete(delete_all=True, namespace=NAMESPACE_NAME)
            except Exception:
                # if namespace not found
                pass

    @pytest.fixture
    def embedding_openai(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings()

    @pytest.fixture
    def texts(self) -> List[str]:
        return ["foo", "bar", "baz"]

    def test_from_texts(
            self, texts: List[str], embedding_openai: OpenAIEmbeddings, get_langchain_pinecone
    ) -> None:
        """Test end to end construction and search."""
        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts.insert(0, needs)

        docsearch = get_langchain_pinecone.PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = docsearch.similarity_search(unique_id, k=1, namespace=NAMESPACE_NAME)
        output[0].id = None  # overwrite ID for ease of comparison
        assert output == [Document(page_content=needs)]

    @pytest.fixture
    def mock_pool_not_supported(self, mocker: MockerFixture) -> None:
        """
        This is the error thrown when multiprocessing is not supported.
        See https://github.com/langchain-ai/langchain/issues/11168
        """
        mocker.patch(
            "multiprocessing.synchronize.SemLock.__init__",
            side_effect=OSError(
                "FileNotFoundError: [Errno 2] No such file or directory"
            ),
        )

    @pytest.mark.usefixtures("mock_pool_not_supported")
    def test_that_async_freq_uses_multiprocessing(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        with pytest.raises(OSError):
            langchain_pinecone.PineconeVectorStore.from_texts(
                texts=texts,
                embedding=embedding_openai,
                index_name=INDEX_NAME,
                namespace=NAMESPACE_NAME,
                async_req=True,
            )

    @pytest.mark.usefixtures("mock_pool_not_supported")
    def test_that_async_freq_false_enabled_singlethreading(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        langchain_pinecone.PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
            async_req=False,
        )
