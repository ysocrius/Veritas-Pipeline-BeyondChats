"""
Tests for data loading and normalization.
"""
import pytest
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval_pipeline.loader import load_data, normalize_conversation, normalize_context


def test_normalize_conversation_mock_format():
    """Test normalizing conversation from mock format (list)."""
    mock_data = [
        {
            "id": "conv_123",
            "messages": [
                {"role": "user", "content": "Hello", "id": "msg_1"},
                {"role": "assistant", "content": "Hi", "id": "msg_2"}
            ]
        }
    ]
    
    result = normalize_conversation(mock_data)
    assert result["id"] == "conv_123"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"


def test_normalize_conversation_assignment_format():
    """Test normalizing conversation from assignment format."""
    assignment_data = {
        "chat_id": 12345,
        "user_id": 67890,
        "conversation_turns": [
            {
                "turn": 1,
                "sender_id": 67890,
                "role": "User",
                "message": "What is IVF?",
                "created_at": "2025-01-01T00:00:00.000000Z"
            },
            {
                "turn": 2,
                "sender_id": 1,
                "role": "AI/Chatbot",
                "message": "IVF is in vitro fertilization.",
                "created_at": "2025-01-01T00:00:05.000000Z"
            }
        ]
    }
    
    result = normalize_conversation(assignment_data)
    assert result["id"] == "12345"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][0]["content"] == "What is IVF?"


def test_normalize_context_mock_format():
    """Test normalizing context from mock format."""
    mock_context = {
        "msg_1": [
            {"text": "Context 1", "vector": [0.1, 0.2], "score": 0.9}
        ]
    }
    
    result = normalize_context(mock_context)
    assert "msg_1" in result
    assert len(result["msg_1"]) == 1


def test_normalize_context_assignment_format():
    """Test normalizing context from assignment format."""
    assignment_context = {
        "status": "success",
        "status_code": 200,
        "message": "Message sent successfully!",
        "data": {
            "vector_data": [
                {
                    "id": 123,
                    "source_url": "https://example.com",
                    "text": "IVF is a fertility treatment.",
                    "tokens": 10,
                    "created_at": "2024-01-01T00:00:00.000Z"
                }
            ]
        }
    }
    
    result = normalize_context(assignment_context)
    assert "context" in result
    assert len(result["context"]) == 1
    assert result["context"][0]["text"] == "IVF is a fertility treatment."


def test_load_data_with_mock_files():
    """Test loading data from mock JSON files."""
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as conv_file:
        conv_data = [{
            "id": "test_conv",
            "messages": [
                {"role": "user", "content": "Test", "id": "msg_1"},
                {"role": "assistant", "content": "Response", "id": "msg_2"}
            ]
        }]
        json.dump(conv_data, conv_file)
        conv_path = conv_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as ctx_file:
        ctx_data = {
            "msg_1": [
                {"text": "Test context", "vector": [0.1], "score": 0.9}
            ]
        }
        json.dump(ctx_data, ctx_file)
        ctx_path = ctx_file.name
    
    try:
        # Load data
        eval_input = load_data(conv_path, ctx_path)
        
        # Verify
        assert eval_input.conversation.id == "test_conv"
        assert len(eval_input.conversation.messages) == 2
        assert "msg_1" in eval_input.context.entries
        
    finally:
        # Cleanup
        Path(conv_path).unlink()
        Path(ctx_path).unlink()


def test_load_data_invalid_json():
    """Test that loading invalid JSON raises an error."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        invalid_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{}")
        valid_path = f.name
    
    try:
        with pytest.raises(ValueError):
            load_data(invalid_path, valid_path)
    finally:
        Path(invalid_path).unlink()
        Path(valid_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
