"""
Tests for target selection logic.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval_pipeline.schemas import Message, Conversation, ContextData, ContextChunk
from eval_pipeline.targeting import select_target_pair


def test_select_target_pair_with_matching_id():
    """Test selecting target pair when message ID matches context key."""
    messages = [
        Message(role="user", content="Query 1", id="msg_u1"),
        Message(role="assistant", content="Response 1", id="msg_a1"),
        Message(role="user", content="Query 2", id="msg_u2"),
        Message(role="assistant", content="Response 2", id="msg_a2")
    ]
    conv = Conversation(id="conv_1", messages=messages)
    
    chunks = [ContextChunk(text="Context for query 2")]
    context = ContextData(entries={"msg_u2": chunks})
    
    result = select_target_pair(conv, context)
    
    assert result is not None
    user_msg, ai_msg, ctx_key = result
    assert user_msg.content == "Query 2"
    assert ai_msg.content == "Response 2"
    assert ctx_key == "msg_u2"


def test_select_target_pair_fallback_to_last():
    """Test fallback to last user-assistant pair when no ID match."""
    messages = [
        Message(role="user", content="Query 1", id="msg_u1"),
        Message(role="assistant", content="Response 1", id="msg_a1"),
        Message(role="user", content="Query 2", id="msg_u2"),
        Message(role="assistant", content="Response 2", id="msg_a2")
    ]
    conv = Conversation(id="conv_1", messages=messages)
    
    # Context with non-matching key
    chunks = [ContextChunk(text="Generic context")]
    context = ContextData(entries={"context": chunks})
    
    result = select_target_pair(conv, context)
    
    assert result is not None
    user_msg, ai_msg, ctx_key = result
    assert user_msg.content == "Query 2"  # Last user message
    assert ai_msg.content == "Response 2"  # Last assistant message
    assert ctx_key == "context"


def test_select_target_pair_no_assistant_after_user():
    """Test when user message has context but no following assistant message."""
    messages = [
        Message(role="user", content="Query 1", id="msg_u1"),
        Message(role="assistant", content="Response 1", id="msg_a1"),
        Message(role="user", content="Query 2", id="msg_u2")
        # No assistant response after msg_u2
    ]
    conv = Conversation(id="conv_1", messages=messages)
    
    chunks = [ContextChunk(text="Context for query 2")]
    context = ContextData(entries={"msg_u2": chunks})
    
    result = select_target_pair(conv, context)
    
    # Should fallback to last complete pair
    assert result is not None
    user_msg, ai_msg, ctx_key = result
    assert user_msg.content == "Query 2"
    assert ai_msg.content == "Response 1"  # Last available assistant message


def test_select_target_pair_empty_conversation():
    """Test with empty conversation."""
    conv = Conversation(id="conv_1", messages=[])
    context = ContextData(entries={"context": [ContextChunk(text="Context")]})
    
    result = select_target_pair(conv, context)
    assert result is None


def test_select_target_pair_no_context():
    """Test with no context entries."""
    messages = [
        Message(role="user", content="Query", id="msg_u1"),
        Message(role="assistant", content="Response", id="msg_a1")
    ]
    conv = Conversation(id="conv_1", messages=messages)
    context = ContextData(entries={})
    
    result = select_target_pair(conv, context)
    assert result is None


def test_select_target_pair_only_user_messages():
    """Test with only user messages (no assistant)."""
    messages = [
        Message(role="user", content="Query 1", id="msg_u1"),
        Message(role="user", content="Query 2", id="msg_u2")
    ]
    conv = Conversation(id="conv_1", messages=messages)
    context = ContextData(entries={"msg_u1": [ContextChunk(text="Context")]})
    
    result = select_target_pair(conv, context)
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
