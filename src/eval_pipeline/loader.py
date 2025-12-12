import json
from pathlib import Path
from typing import Tuple, Dict, List, Any
from .schemas import Conversation, ContextData, EvalInput, Message, ContextChunk

# Try to import json5 for lenient parsing (handles trailing commas)
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False

def load_json(path: str) -> dict:
    """
    Load JSON with lenient parsing to handle common formatting issues.
    Uses json5 library if available to handle trailing commas and other
    common JSON formatting issues found in real-world data.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try standard JSON first (fastest)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # If standard parsing fails and json5 is available, use it
        if HAS_JSON5:
            try:
                return json5.loads(content)
            except Exception:
                # json5 also failed, try aggressive cleanup
                pass
        
        # Aggressive cleanup for badly formatted JSON
        import re
        # Remove trailing commas
        fixed_content = re.sub(r',\s*([}\]])', r'\1', content)
        
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError as e:
            # Last resort: provide helpful error message
            raise ValueError(
                f"Unable to parse JSON from {path}. "
                f"Error: {e}. "
                f"The file may have formatting issues beyond trailing commas. "
                f"Please verify the JSON is valid."
            )

def normalize_conversation(conv_raw: Any) -> Dict:
    """
    Normalizes conversation data from different formats to our standard schema.
    Handles both mock format and assignment sample format.
    """
    # Handle list format (mock data)
    if isinstance(conv_raw, list):
        if len(conv_raw) > 0:
            conv_data = conv_raw[0]
        else:
            raise ValueError("Conversation JSON list is empty")
    else:
        conv_data = conv_raw
    
    # Check if this is assignment format (has 'conversation_turns' instead of 'messages')
    if 'conversation_turns' in conv_data:
        # Convert assignment format to our schema format
        messages = []
        for turn in conv_data['conversation_turns']:
            # Map assignment format to our Message schema
            role = 'user' if turn['role'].lower() == 'user' else 'assistant'
            msg = {
                'role': role,
                'content': turn['message'],
                'id': f"turn_{turn['turn']}",  # Generate ID from turn number
                'timestamp': None  # Could parse created_at if needed
            }
            messages.append(msg)
        
        return {
            'id': str(conv_data.get('chat_id', 'unknown')),
            'messages': messages
        }
    
    # Already in our format
    return conv_data

def normalize_context(context_raw: Any) -> Dict[str, List[Dict]]:
    """
    Normalizes context data from different formats to our standard schema.
    Handles both mock format and assignment sample format.
    """
    # Check if this is assignment format (has 'data' wrapper with 'vector_data')
    if isinstance(context_raw, dict) and 'data' in context_raw:
        vector_data = context_raw['data'].get('vector_data', [])
        
        # Since assignment format doesn't have message_id mapping,
        # we'll create a generic key that can be matched later
        chunks = []
        for item in vector_data:
            chunk = {
                'text': item.get('text', ''),
                'vector': item.get('vector', []),
                'score': item.get('score')
            }
            chunks.append(chunk)
        
        # Return with a generic key - targeting logic will handle this
        return {'context': chunks}
    
    # Already in our format (dict with message_id keys)
    return context_raw

def load_data(conversation_path: str, context_path: str) -> EvalInput:
    """
    Loads conversation and context data from JSON files and validates them against schemas.
    Handles multiple input formats (mock data and assignment samples).
    """
    try:
        conv_raw = load_json(conversation_path)
        context_raw = load_json(context_path)
        
        # Normalize to our schema format
        conv_data = normalize_conversation(conv_raw)
        context_data = normalize_context(context_raw)
        
        # Validate against Pydantic schemas
        conversation = Conversation(**conv_data)
        context = ContextData(entries=context_data)

        return EvalInput(conversation=conversation, context=context)
    
    except Exception as e:
        raise ValueError(f"Failed to load or validate input data: {e}")
