"""
Tests for data schemas and validation.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval_pipeline.schemas import Message, Conversation, ContextChunk, ContextData, EvalInput


def test_message_creation():
    """Test creating a valid Message."""
    msg = Message(role="user", content="Hello", id="msg_1", timestamp=1234567890.0)
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.id == "msg_1"


def test_message_minimal():
    """Test creating a Message with minimal required fields."""
    msg = Message(role="assistant", content="Hi there")
    assert msg.role == "assistant"
    assert msg.content == "Hi there"
    assert msg.id is None
    assert msg.timestamp is None


def test_conversation_creation():
    """Test creating a valid Conversation."""
    messages = [
        Message(role="user", content="What is AI?", id="msg_1"),
        Message(role="assistant", content="AI is artificial intelligence.", id="msg_2")
    ]
    conv = Conversation(id="conv_1", messages=messages)
    assert conv.id == "conv_1"
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"


def test_context_chunk_creation():
    """Test creating a valid ContextChunk."""
    chunk = ContextChunk(
        text="Paris is the capital of France.",
        vector=[0.1, 0.2, 0.3],
        score=0.95
    )
    assert chunk.text == "Paris is the capital of France."
    assert len(chunk.vector) == 3
    assert chunk.score == 0.95


def test_context_chunk_minimal():
    """Test creating a ContextChunk with minimal fields."""
    chunk = ContextChunk(text="Some context")
    assert chunk.text == "Some context"
    assert chunk.vector == []
    assert chunk.score is None


def test_context_data_creation():
    """Test creating valid ContextData."""
    chunks = [
        ContextChunk(text="Context 1", vector=[0.1, 0.2]),
        ContextChunk(text="Context 2", vector=[0.3, 0.4])
    ]
    context = ContextData(entries={"msg_1": chunks})
    assert "msg_1" in context.entries
    assert len(context.entries["msg_1"]) == 2


def test_eval_input_creation():
    """Test creating a complete EvalInput."""
    messages = [
        Message(role="user", content="Test query", id="msg_1"),
        Message(role="assistant", content="Test response", id="msg_2")
    ]
    conv = Conversation(id="conv_1", messages=messages)
    
    chunks = [ContextChunk(text="Test context")]
    context = ContextData(entries={"msg_1": chunks})
    
    eval_input = EvalInput(conversation=conv, context=context)
    assert eval_input.conversation.id == "conv_1"
    assert "msg_1" in eval_input.context.entries


def test_invalid_message_missing_role():
    """Test that Message validation fails without role."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        Message(content="Hello")


def test_invalid_message_missing_content():
    """Test that Message validation fails without content."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        Message(role="user")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
