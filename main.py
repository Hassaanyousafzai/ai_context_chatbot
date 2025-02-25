import os
import re
import uuid
import time
import datetime
import logging
import functools
from dotenv import load_dotenv
from nltk.corpus import stopwords
import nltk
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

collection_name = "chat_history"
vector_size = 384
distance_metric = models.Distance.COSINE

# Check if the collection exists
try:
    qdrant_client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception as e:
    # If the collection does not exist, create it
    print(f"Collection '{collection_name}' does not exist. Creating a new collection.")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=distance_metric,
        )
    )
    print(f"Collection '{collection_name}' created successfully.")

# Configure the Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize a simple LRU cache for context retrieval (Performance optimization)
context_cache = {}
MAX_CACHE_SIZE = 100

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def format_timestamp(timestamp_str):
    """Format ISO timestamp to a more human-readable format"""
    try:
        timestamp = datetime.datetime.fromisoformat(timestamp_str)
        return timestamp.strftime("%Y-%m-%d at %H:%M:%S")
    except Exception as e:
        return timestamp_str  # Return original if parsing fails

# Add simple caching decorator for performance improvement
def cache_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a simple cache key from arguments
        cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
        
        if cache_key in context_cache:
            return context_cache[cache_key]
        
        result = func(*args, **kwargs)
        
        # Store in cache (with simple LRU implementation)
        if len(context_cache) >= MAX_CACHE_SIZE:
            # Remove oldest item (simple approach)
            context_cache.pop(next(iter(context_cache)))
        
        context_cache[cache_key] = result
        return result
    
    return wrapper

def store_user_message(user_message, user_id):
    """Store user message with embedding in Qdrant"""
    try:
        user_embedding = embedding_model.encode(user_message).tolist()
        
        # Create a unique ID for the message
        message_id = str(uuid.uuid4())
        
        # Prepare the data point with metadata
        point = PointStruct(
            id=message_id,
            vector=user_embedding,
            payload={
                "user_id": user_id,
                "message": user_message,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "user_message"
            }
        )
        
        # Upsert the data point into the collection
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        return message_id
    except Exception as e:
        print(f"Error storing message: {str(e)}")
        return None

@cache_result
def retrieve_context(user_message, user_id, top_k=5):
    """Retrieve relevant context based on semantic similarity"""
    try:
        query_embedding = embedding_model.encode(user_message).tolist()
        
        # Search for similar messages in the collection
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=top_k
        )
        
        # Extract the relevant messages and their timestamps
        context = [
            (hit.payload["message"], hit.payload["timestamp"], hit.score)
            for hit in search_result
            if hit.score > 0.4
        ]
        
        return context
    
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return []  # Return empty context in case of error

@cache_result
def retrieve_topic_history(topic_keywords, user_id, top_k=5):
    """Retrieve specific topic history using keywords"""
    try:
        # Create a combined query string from keywords
        query = " ".join(topic_keywords)
        
        # Generate embedding for the topic query
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search for related messages
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=top_k
        )
        
        # Extract the relevant messages with timestamps
        topic_history = [
            (hit.payload["message"], hit.payload["timestamp"], hit.score)
            for hit in search_result
        ]
        
        return topic_history
    except Exception as e:
        print(f"Error retrieving topic history: {str(e)}")
        return []

def extract_topics(user_message):
    """Extract potential topics from user message for targeted retrieval with improved NLP"""
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase and tokenize by non-alphanumeric chars
    words = re.findall(r'\b[a-z0-9]+\b', user_message.lower())
    
    # Filter out stop words and short words
    topics = [word for word in words if word not in stop_words and len(word) > 2]
    
    phrases = []
    if len(user_message.split()) >= 3:
        words_list = user_message.lower().split()
        for i in range(len(words_list) - 1):
            if words_list[i] not in stop_words and words_list[i+1] not in stop_words:
                phrases.append(f"{words_list[i]} {words_list[i+1]}")
    
    all_topics = topics + phrases
    
    return sorted(all_topics, key=len, reverse=True)[:10]

def is_explicit_history_query(user_message):
    """Determine if this is an explicit question about past interactions"""
    history_indicators = [
        "when did i", "what time did i", "last time i", 
        "previously i", "earlier i", "remember when i", 
        "did i mention", "have i talked about", "when was the last time i",
        "do you remember", "have we discussed", "did we talk about",
        "what did i say about", "when did i last mention", "what was my last",
        "what was i", "did i tell you about"
    ]
    
    lower_message = user_message.lower()
    return any(indicator in lower_message for indicator in history_indicators)

def is_followup_question(message):
    """Detect if a message is likely a follow-up question"""
    followup_indicators = [
        "how was it", "how was that", "how did it go", 
        "how did that go", "and then", "what happened", 
        "tell me more", "what about", "and how was", 
        "what did you think", "was it good", "did you enjoy",
        "why", "how come", "and", "but", "so", "then",
        "what else", "anything else"
    ]
    
    # Check if message is short (typical of follow-ups)
    is_short = len(message.split()) < 6

    lower_message = message.lower()
    has_indicator = any(indicator in lower_message for indicator in followup_indicators)
 
    has_question_mark = "?" in message and is_short
    
    return is_short or has_indicator or has_question_mark

def chat_with_gemini(user_message, user_id, recent_conversation=None):
    """Generate contextual response using GEMINI and retrieved context"""
    if recent_conversation is None:
        recent_conversation = []
    
    try:
        # Store the current user message
        message_id = store_user_message(user_message, user_id)
        logger.info(f"Stored message with ID: {message_id}")
        
        topics = extract_topics(user_message)

        is_history_query = is_explicit_history_query(user_message)
        
        is_followup = is_followup_question(user_message)
        
        if is_history_query and topics:
            logger.info(f"History query detected with topics: {topics}")
            context = retrieve_topic_history(topics, user_id, top_k=5)
        else:
            # Otherwise get general context
            context = retrieve_context(user_message, user_id)
        
        formatted_historical_context = ""
        if context:
            formatted_historical_context = "Previous relevant conversations:\n"
            for idx, (msg, ts, score) in enumerate(context, 1):
                readable_ts = format_timestamp(ts)
                formatted_historical_context += f"{idx}. User said: \"{msg}\" (on {readable_ts}, relevance: {score:.2f})\n"
        else:
            formatted_historical_context = "No relevant historical context found."
        
        # Add current conversation context
        current_conversation_context = ""
        if recent_conversation:
            current_conversation_context = "\nCurrent conversation thread:\n"
            for idx, (role, msg) in enumerate(recent_conversation, 1):
                current_conversation_context += f"{idx}. {role}: \"{msg}\"\n"
        
        # Combine both contexts
        combined_context = formatted_historical_context + current_conversation_context
        logger.info("Context preparation complete")
        
        system_instruction = create_system_instruction(combined_context, is_history_query, not bool(context))
        
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_instruction
        )
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        response = model.generate_content(
            user_message,
            generation_config=generation_config
        )
        
        response_text = response.text
        logger.info("Generated response successfully")
        
        recent_conversation.append(("User", user_message))
        recent_conversation.append(("Assistant", response_text))
        
        if len(recent_conversation) > 10:
            recent_conversation = recent_conversation[-10:]
        
        return response_text, recent_conversation
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        error_response = "I apologize, but I'm having trouble generating a response. Please try again."
        if recent_conversation is not None:
            recent_conversation.append(("User", user_message))
            recent_conversation.append(("Assistant", error_response))
        return error_response, recent_conversation

def create_system_instruction(context, is_history_query=False, no_context=False):
    """Create system instruction with embedded context and additional handling"""
    
    base_instruction = """
    You are a helpful assistant who is an expert at conversing with the user about their day-to-day life and activities.
    """
    
    context_section = f"""
    The following is relevant context from past conversations with this user:
    {context}
    """
    
    # Guidelines with specific handling for different scenarios
    guidelines = """
    Guidelines for your responses:
    1. Only explicitly reference past conversations when directly asked about them (e.g., "Did I mention X?", "When did I talk about Y?").
    2. When specifically asked about past events, include the exact date and time from the context like "Last time you mentioned [topic] was on [date] at [time]."
    3. MAINTAIN CONVERSATIONAL CONTEXT from the current conversation thread. If the user mentions something like "I played football and it was amazing" and later asks "how was the football game?", remember the details they shared about it being amazing.
    4. If the user asks a follow-up question like "and how was it?" or "how was the game?", connect it to the most recently discussed topic rather than asking for clarification.
    5. For questions like "What did I say about my plans?" or "What was my last project about?", respond with the exact information the user shared, e.g., "You said you plan to visit the park tomorrow."
    6. Track the current conversation's topic flow to handle ambiguous follow-up questions intelligently.
    7. Never invent or hallucinate details that aren't explicitly in the context or the current conversation.
    8. When the user asks about details of something they mentioned earlier in the CURRENT conversation, use that information without prompting them to repeat it.
    """
    
    if is_history_query and no_context:
        guidelines += """
        9. Since this appears to be a question about past conversations but no relevant context was found, clearly inform the user: "I don't have any record of you mentioning [topic]."
        """
    elif no_context:
        guidelines += """
        9. Though no specific historical context was found, focus on maintaining the flow of the current conversation.
        """
    
    return base_instruction + context_section + guidelines

if __name__ == "__main__":
    user_id = "123"
    recent_conversation = []
    print("Start chatting with the assistant (type 'exit' to stop):")
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
                
            start_time = time.time()
            bot_response, recent_conversation = chat_with_gemini(user_input, user_id, recent_conversation)
            elapsed_time = time.time() - start_time
            
            print(f"\nAssistant: {bot_response}")
            print(f"[Response time: {elapsed_time:.2f}s]")
    except KeyboardInterrupt:
        print("\nExiting chatbot...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")