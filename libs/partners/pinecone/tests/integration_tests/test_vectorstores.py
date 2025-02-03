import importlib
import os
import sys
import time
import uuid
from importlib import reload
from typing import List

import numpy as np
import pinecone  # type: ignore
import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests, EMBEDDING_SIZE
from pytest_mock import MockerFixture  # type: ignore[import-not-found]

INDEX_NAME = "langchain-test-index"  # name of the index

DEFAULT_SLEEP = 10

def add_delay(delay_seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            time.sleep(delay_seconds)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class TestPinecone(VectorStoreIntegrationTests):
    index: "pinecone.grpc.GRPCIndex"
    pc: "pinecone.grpc.PineconeGRPC"

    def setup_class(cls) -> None:
        from pinecone import ServerlessSpec
        from pinecone.grpc import PineconeGRPC as PineconeClient

        client = PineconeClient(api_key=os.environ["PINECONE_API_KEY"])
        if INDEX_NAME in client.list_indexes().names():
            client.delete_index(INDEX_NAME)
            while not client.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_SIZE,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        cls.index = client.Index(INDEX_NAME)
        cls.client = client

    @classmethod
    def teardown_class(cls) -> None:
        cls.client.delete_index(INDEX_NAME)

    @pytest.fixture(scope="class", params=[False, True], ids=["http", "grpc"])
    def langchain_pinecone(self, request, class_mocker: MockerFixture):
        if not request.param:
            class_mocker.patch.dict(sys.modules, {"pinecone.grpc": None})
        import langchain_pinecone
        return langchain_pinecone

    @pytest.fixture
    def vectorstore(self, request, mocker: MockerFixture, langchain_pinecone) -> VectorStore:
        for attr_name, attr_value in langchain_pinecone.PineconeVectorStore.__dict__.items():
            if callable(attr_value):
                attr_wrapped = add_delay(DEFAULT_SLEEP)(attr_value)
                mocker.patch.object(langchain_pinecone.PineconeVectorStore, attr_name, attr_wrapped)

        namespace = uuid.uuid4().hex # Just use a new namespace for each test instead of cleaning up after each one
        return langchain_pinecone.PineconeVectorStore(embedding=self.get_embeddings(), index_name=INDEX_NAME, namespace=namespace)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    def test_get_by_ids(self, vectorstore: VectorStore) -> None:
        super().test_get_by_ids(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    def test_get_by_ids_missing(self, vectorstore: VectorStore) -> None:
        super().test_get_by_ids_missing(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    def test_add_documents_documents(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_documents(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_with_existing_ids(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    async def test_get_by_ids_async(self, vectorstore: VectorStore) -> None:
        await super().test_get_by_ids_async(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    async def test_get_by_ids_missing_async(self, vectorstore: VectorStore) -> None:
        await super().test_get_by_ids_missing_async(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    async def test_add_documents_documents_async(self, vectorstore: VectorStore) -> None:
        await super().test_add_documents_documents_async(vectorstore)

    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
    async def test_add_documents_with_existing_ids_async(self, vectorstore: VectorStore) -> None:
        await super().test_add_documents_with_existing_ids_async(vectorstore)