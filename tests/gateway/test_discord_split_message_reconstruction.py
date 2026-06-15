"""Tests for split-message reconstruction through the require_mention gate.

When a peer's Discord message exceeds 2000 chars, Discord fragments it into
multiple messages: only part 1 carries the @mention; parts 2+ are plain text.
The require_mention gate used to drop parts 2+, giving the agent an incomplete
turn.  The fix: if a pending text batch already exists for the session (meaning
part 1 passed auth and was queued), continuation parts bypass the gate.

Covers:
- _has_pending_split_batch: returns True iff a pending batch exists for the session
- End-to-end: mention in part 1, no mention in parts 2+, all parts concatenated
- Regression: a truly unrelated message (no pending batch, no mention) is still dropped
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(require_mention: bool = True, group_sessions_per_user: bool = True):
    """Minimal DiscordAdapter wired for split-message tests."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    config.extra = {
        "group_sessions_per_user": group_sessions_per_user,
        "thread_sessions_per_user": False,
        "require_mention": require_mention,
    }
    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


def _make_event(
    text: str,
    chat_id: str = "111222333",
    user_id: str = "999888777",
    chat_type: str = "group",
    thread_id: str = None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            thread_id=thread_id,
        ),
    )


def _make_discord_message(
    content: str,
    channel_id: str = "111222333",
    author_id: str = "999888777",
    mentioned_bot_id: str = None,
    is_dm: bool = False,
    is_thread: bool = False,
    thread_id: str = None,
):
    """Build a minimal fake discord.Message for testing.

    We do NOT use spec= on channel mocks because when discord.py is
    conditionally imported (DISCORD_AVAILABLE check), the channel classes
    (TextChannel, Thread, DMChannel) resolve as MagicMock objects rather than
    real classes, causing InvalidSpecError.  Plain MagicMock() with the
    right attributes is sufficient.
    """
    import discord

    author = MagicMock()
    author.id = int(author_id)
    author.bot = True  # simulates inter-bot message
    author.display_name = "TestBot"

    bot_user = MagicMock()
    bot_user.id = int(mentioned_bot_id) if mentioned_bot_id else 111

    if is_dm:
        channel = MagicMock()
        channel.__class__ = discord.DMChannel
        channel.id = int(channel_id)
    elif is_thread:
        channel = MagicMock()
        channel.__class__ = discord.Thread
        channel.id = int(thread_id or channel_id)
        channel.parent_id = int(channel_id)
    else:
        channel = MagicMock()
        # Not a DMChannel or Thread — isinstance checks will return False
        channel.__class__ = type("TextChannel", (), {})
        channel.id = int(channel_id)

    msg = MagicMock()
    msg.content = content
    msg.author = author
    msg.channel = channel
    if mentioned_bot_id and int(mentioned_bot_id) == bot_user.id:
        msg.mentions = [bot_user]
    else:
        msg.mentions = []

    return msg, bot_user


# ---------------------------------------------------------------------------
# Unit tests for _has_pending_split_batch
# ---------------------------------------------------------------------------

class TestHasPendingSplitBatch:
    def test_no_pending_batch_returns_false(self):
        adapter = _make_adapter()
        msg, _ = _make_discord_message("hello", channel_id="111", author_id="999")
        assert adapter._has_pending_split_batch(msg, thread_id=None) is False

    def test_pending_batch_for_same_session_returns_true(self):
        adapter = _make_adapter()
        # Seed a pending batch for channel 111, user 999
        event = _make_event("part 1", chat_id="111", user_id="999")
        key = adapter._text_batch_key(event)
        adapter._pending_text_batches[key] = event

        msg, _ = _make_discord_message("part 2", channel_id="111", author_id="999")
        assert adapter._has_pending_split_batch(msg, thread_id=None) is True

    def test_pending_batch_different_user_returns_false(self):
        adapter = _make_adapter()
        event = _make_event("part 1", chat_id="111", user_id="999")
        key = adapter._text_batch_key(event)
        adapter._pending_text_batches[key] = event

        # Different author — different session key
        msg, _ = _make_discord_message("part 2", channel_id="111", author_id="888")
        assert adapter._has_pending_split_batch(msg, thread_id=None) is False

    def test_pending_batch_different_channel_returns_false(self):
        adapter = _make_adapter()
        event = _make_event("part 1", chat_id="111", user_id="999")
        key = adapter._text_batch_key(event)
        adapter._pending_text_batches[key] = event

        # Different channel — different session key
        msg, _ = _make_discord_message("part 2", channel_id="222", author_id="999")
        assert adapter._has_pending_split_batch(msg, thread_id=None) is False

    def test_pending_batch_thread(self):
        adapter = _make_adapter()
        # Thread sessions share across users (thread_sessions_per_user=False)
        event = _make_event("part 1", chat_id="555", user_id="999",
                             chat_type="thread", thread_id="777")
        key = adapter._text_batch_key(event)
        adapter._pending_text_batches[key] = event

        msg, _ = _make_discord_message("part 2", channel_id="555",
                                        is_thread=True, thread_id="777")
        assert adapter._has_pending_split_batch(msg, thread_id="777") is True


# ---------------------------------------------------------------------------
# End-to-end batching: mention in part 1, no mention in parts 2+
# ---------------------------------------------------------------------------

class TestSplitMessageReconstructionE2E:
    @pytest.mark.asyncio
    async def test_two_part_split_full_text_reaches_agent(self):
        """Part 1 mentions bot; part 2 does not. Agent receives both parts."""
        adapter = _make_adapter()

        # Part 1: has mention, queued via _enqueue_text_event
        part1 = _make_event(
            "x" * 1990,  # near the 2000-char limit
            chat_id="111", user_id="999",
        )
        adapter._enqueue_text_event(part1)

        # Part 2: no mention — would have been dropped before the fix
        part2 = _make_event("continuation text", chat_id="111", user_id="999")
        adapter._enqueue_text_event(part2)

        # Flush
        await asyncio.sleep(0.5)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert "continuation text" in dispatched.text
        assert len(dispatched.text) > 1990

    @pytest.mark.asyncio
    async def test_three_part_split_all_parts_reach_agent(self):
        adapter = _make_adapter()

        for text in ["part A — " + "x" * 1980, "part B", "part C"]:
            event = _make_event(text, chat_id="111", user_id="999")
            adapter._enqueue_text_event(event)
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.5)

        adapter.handle_message.assert_called_once()
        full_text = adapter.handle_message.call_args[0][0].text
        assert "part A" in full_text
        assert "part B" in full_text
        assert "part C" in full_text

    @pytest.mark.asyncio
    async def test_unrelated_message_no_pending_batch_not_passed(self):
        """A message with no mention AND no pending batch is still rejected.

        This test verifies the gate holds for truly unauthorised messages —
        the bypass only applies when a batch is already pending (part 1 validated).
        """
        adapter = _make_adapter()
        # _pending_text_batches is empty — no prior part 1
        assert len(adapter._pending_text_batches) == 0

        # _has_pending_split_batch should return False for a fresh channel/user
        msg, _ = _make_discord_message("hello from stranger", channel_id="111", author_id="999")
        assert adapter._has_pending_split_batch(msg, thread_id=None) is False

    @pytest.mark.asyncio
    async def test_batch_cleared_after_flush_does_not_pass_new_unmentioned_message(self):
        """After the batch flushes, a new mention-free message is rejected again."""
        adapter = _make_adapter()

        part1 = _make_event("x" * 100, chat_id="111", user_id="999")
        adapter._enqueue_text_event(part1)

        # Flush
        await asyncio.sleep(0.2)
        adapter.handle_message.assert_called_once()

        # Batch is now gone
        assert len(adapter._pending_text_batches) == 0

        # A new mention-free message should NOT have a pending batch to pass through
        msg, _ = _make_discord_message("new message without mention",
                                        channel_id="111", author_id="999")
        assert adapter._has_pending_split_batch(msg, thread_id=None) is False
