from typing import Tuple, Optional
from .schemas import Conversation, Message, ContextData

def select_target_pair(conversation: Conversation, context: ContextData) -> Optional[Tuple[Message, Message, str]]:
    """
    Identifies the (User_Message, AI_Response, Context_Key) triplet to evaluate.
    
    Strategy:
    1. Look for a User message ID that exists in the ContextData keys.
    2. If found, pair it with the immediately following Assistant message.
    3. Fallback: Use last user-assistant pair with available context.
    """
    
    messages = conversation.messages
    
    # Strategy 1: Try to match message IDs with context keys
    for i, msg in enumerate(messages):
        if msg.role == 'user':
            if msg.id and msg.id in context.entries:
                # Found a user message with context.
                # Check if next message is assistant
                if i + 1 < len(messages) and messages[i+1].role == 'assistant':
                    return (msg, messages[i+1], msg.id)
    
    # Strategy 2: Fallback - find last meaningful user-assistant exchange
    # This handles cases where context is generic or IDs don't match
    last_user = None
    last_assistant = None
    
    for msg in messages:
        if msg.role == 'user':
            last_user = msg
        elif msg.role == 'assistant':
            # Only update if we have a user message before this
            if last_user:
                last_assistant = msg
            
    if last_user and last_assistant:
        # Use first available context key (often 'context' for assignment samples)
        first_context_key = next(iter(context.entries)) if context.entries else None
        if first_context_key:
            return (last_user, last_assistant, first_context_key)

    return None
