"""Concurrency guarantees for request-scoped and global providers.

These tests pin down behavior observed during a production incident where
request-scoped dependencies leaked ("context bleed") between two concurrent
requests. Synchronization uses ``asyncio`` / ``threading`` primitives (events)
instead of ``sleep`` so a regression fails deterministically rather than
flaking on timing.
"""

import asyncio
import threading
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from that_depends import BaseContainer, ContextScopes, providers
from that_depends.providers import container_context


def create_request_resource() -> typing.Iterator[typing.Any]:
    yield object()


async def create_async_request_resource() -> typing.AsyncIterator[typing.Any]:
    yield object()


class RequestContainer(BaseContainer):
    """Container with request-scoped resources."""

    alias = "concurrency_request_container"
    default_scope = ContextScopes.REQUEST
    request_resource = providers.ContextResource(create_request_resource)
    async_request_resource = providers.ContextResource(create_async_request_resource)


@pytest.fixture(autouse=True)
async def _tear_down_container() -> typing.AsyncIterator[None]:
    try:
        yield
    finally:
        RequestContainer.reset_override_sync()
        await RequestContainer.tear_down()


async def test_request_scoped_resources_are_independent_between_concurrent_requests() -> None:
    """Two overlapping requests must never share a request-scoped instance."""
    resolved: dict[str, typing.Any] = {}
    a_resolved = asyncio.Event()
    b_can_finish = asyncio.Event()

    async def request_a() -> None:
        async with container_context(RequestContainer, scope=ContextScopes.REQUEST):
            resolved["a"] = await RequestContainer.async_request_resource.resolve()
            a_resolved.set()
            await b_can_finish.wait()
            # A stays inside its own context after B is fully torn down:
            # re-resolving must return A's original instance, not B's.
            assert await RequestContainer.async_request_resource.resolve() is resolved["a"]

    async def request_b() -> None:
        await a_resolved.wait()
        async with container_context(RequestContainer, scope=ContextScopes.REQUEST):
            resolved["b"] = await RequestContainer.async_request_resource.resolve()
        # B has torn its context down before A is allowed to finish.

    task_a = asyncio.create_task(request_a())
    task_b = asyncio.create_task(request_b())
    await task_b
    b_can_finish.set()
    await asyncio.gather(task_a, task_b)

    assert resolved["a"] is not resolved["b"]


def test_request_scoped_resources_are_independent_between_concurrent_sync_requests() -> None:
    """The sync twin of the request-isolation guarantee, exercised on threads."""
    resolved: dict[str, typing.Any] = {}
    a_resolved = threading.Event()
    b_can_finish = threading.Event()

    def request_a() -> None:
        with container_context(RequestContainer, scope=ContextScopes.REQUEST):
            resolved["a"] = RequestContainer.request_resource.resolve_sync()
            a_resolved.set()
            b_can_finish.wait()
            assert RequestContainer.request_resource.resolve_sync() is resolved["a"]

    def request_b() -> None:
        a_resolved.wait()
        with container_context(RequestContainer, scope=ContextScopes.REQUEST):
            resolved["b"] = RequestContainer.request_resource.resolve_sync()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(request_a)
        future_b = pool.submit(request_b)
        future_b.result()
        b_can_finish.set()
        future_a.result()

    assert resolved["a"] is not resolved["b"]


async def test_singleton_is_initialized_once_under_concurrent_resolve() -> None:
    """A global singleton factory must run exactly once for concurrent resolves."""
    calls = 0
    entered_factory = asyncio.Event()
    release_factory = asyncio.Event()

    async def create_instance() -> object:
        nonlocal calls
        calls += 1
        entered_factory.set()
        await release_factory.wait()
        return object()

    singleton = providers.AsyncSingleton(create_instance)

    async def resolve() -> typing.Any:
        return await singleton.resolve()

    first = asyncio.create_task(resolve())
    await entered_factory.wait()
    # The factory is suspended mid-creation; the second resolve must wait and
    # reuse the in-flight instance instead of running the factory again.
    second = asyncio.create_task(resolve())
    await asyncio.sleep(0)
    assert calls == 1

    release_factory.set()
    first_instance, second_instance = await asyncio.gather(first, second)

    assert first_instance is second_instance
    assert calls == 1


def test_sync_singleton_is_initialized_once_under_concurrent_resolve() -> None:
    """The sync singleton factory must run exactly once across threads."""
    calls = 0
    entered_factory = threading.Event()
    release_factory = threading.Event()

    def create_instance() -> object:
        nonlocal calls
        calls += 1
        entered_factory.set()
        release_factory.wait()
        return object()

    singleton = providers.Singleton(create_instance)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(singleton.resolve_sync)
        assert entered_factory.wait(timeout=5)
        second = pool.submit(singleton.resolve_sync)
        release_factory.set()
        first_instance = first.result(timeout=5)
        second_instance = second.result(timeout=5)

    assert first_instance is second_instance
    assert calls == 1


async def test_async_generator_resource_is_finalized_when_request_is_cancelled() -> None:
    """Cancelling a request must run the async generator's cleanup block."""
    entered = asyncio.Event()
    finalized = asyncio.Event()

    async def create_cancellable_resource() -> typing.AsyncIterator[str]:
        entered.set()
        try:
            yield uuid.uuid4().hex
        finally:
            finalized.set()

    resource = providers.ContextResource(create_cancellable_resource)

    async def handle_request() -> None:
        async with container_context(resource, scope=ContextScopes.REQUEST):
            await resource.resolve()
            await asyncio.Event().wait()

    task = asyncio.create_task(handle_request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finalized.is_set()


async def test_cancelled_creation_is_cleaned_up_and_waiting_request_rebuilds() -> None:
    """Cancelling mid-creation finalizes the generator and lets a waiter rebuild.

    One request suspends inside the resource creator while a second request is
    queued on the resolve lock. Cancelling the first request must finalize its
    half-built generator, and the queued request must observe a fresh instance
    rather than a poisoned/empty context.
    """
    entered = asyncio.Event()
    release_creator = asyncio.Event()
    creations = 0
    finalizations = 0

    async def create_resource() -> typing.AsyncIterator[str]:
        nonlocal creations
        creations += 1
        entered.set()
        try:
            await release_creator.wait()
            yield uuid.uuid4().hex
        finally:
            nonlocal finalizations
            finalizations += 1

    resource = providers.ContextResource(create_resource)

    # Both tasks share one request context (e.g. a request that fans out
    # concurrently); the second task queues on the same resolve lock.
    async with container_context(resource, scope=ContextScopes.REQUEST):
        cancelled = asyncio.create_task(resource.resolve())
        await entered.wait()
        waiting = asyncio.create_task(resource.resolve())
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release_creator.set()
        value = await waiting

    assert value
    assert creations == 2
    assert finalizations == 2


async def test_request_resource_is_recreated_in_a_following_request_after_cancel() -> None:
    """Cleanup on cancellation must not poison the next request's context."""
    creations = 0
    finalizations = 0
    entered = asyncio.Event()

    async def create_resource() -> typing.AsyncIterator[str]:
        nonlocal creations
        creations += 1
        try:
            entered.set()
            yield uuid.uuid4().hex
        finally:
            nonlocal finalizations
            finalizations += 1

    resource = providers.ContextResource(create_resource)

    async def cancelled_request() -> None:
        async with container_context(resource, scope=ContextScopes.REQUEST):
            await resource.resolve()
            await asyncio.Event().wait()

    task = asyncio.create_task(cancelled_request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with container_context(resource, scope=ContextScopes.REQUEST):
        value = await resource.resolve()

    assert value
    assert creations == 2
    assert finalizations == 2
