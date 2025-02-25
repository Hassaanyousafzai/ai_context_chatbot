import streamlit as st
import time
from main import chat_with_gemini, store_user_message

def main():
    st.title("Memory-Enabled Chat Assistant")

    if "app_loaded" not in st.session_state:
        with st.spinner("Loading Chat Assistant..."):
            time.sleep(2)
        st.session_state.app_loaded = True
        st.rerun()

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "streamlit_user_123"

    if 'recent_conversation' not in st.session_state:
        st.session_state.recent_conversation = []

    if 'trigger_query' not in st.session_state:
        st.session_state.trigger_query = None
    
    for message in st.session_state.conversation_history:
        role = message[0].lower()
        content = message[1]
        with st.chat_message(role):
            st.write(content)

    with st.sidebar:
        st.subheader("Test Queries")
        st.write("Try these test sequences:")
        
        if st.button("Test: Memory Retention"):
            st.session_state.trigger_query = "memory_retention"
            st.rerun()
        
        if st.button("Test: Contextual Query"):
            st.session_state.trigger_query = "contextual_query"
            st.rerun()
        
        if st.button("Test: Semantic Search"):
            st.session_state.trigger_query = "semantic_search"
            st.rerun()
        
        if st.button("Test: Error Handling"):
            st.session_state.trigger_query = "error_handling"
            st.rerun()
        
        if st.button("Test: Cross-Session Memory"):
            st.session_state.trigger_query = "cross_session"
            st.rerun()

    if st.session_state.trigger_query:
        if st.session_state.trigger_query == "memory_retention":
            process_message("I am going to drink water.")
            time.sleep(1)
            process_message("When did I last drink water?")
        
        elif st.session_state.trigger_query == "contextual_query":
            process_message("I plan to visit the park tomorrow.")
            time.sleep(1)
            process_message("What did I say about my plans?")
        
        elif st.session_state.trigger_query == "semantic_search":
            process_message("I will read a book tonight.")
            time.sleep(1)
            process_message("When did I last mention reading?")
        
        elif st.session_state.trigger_query == "error_handling":
            process_message("What did I say about Mars?")
        
        elif st.session_state.trigger_query == "cross_session":
            process_message("I am working on a project about AI.")
            time.sleep(1)
            st.session_state.recent_conversation = []
            process_message("What was my last project about?")
        
        st.session_state.trigger_query = None
        st.rerun()

    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        process_message(user_input)

def process_message(user_input):
    st.session_state.conversation_history.append(("User", user_input))
    
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("Thinking...")
        
        start_time = time.time()
        response, updated_conversation = chat_with_gemini(
            user_input, 
            st.session_state.user_id, 
            st.session_state.recent_conversation
        )
        elapsed_time = time.time() - start_time

        st.session_state.recent_conversation = updated_conversation
        
        message_placeholder.write(response)
        st.caption(f"⏱ Response time: {elapsed_time:.2f}s")
    
    st.session_state.conversation_history.append(("Assistant", response))

if __name__ == "__main__":
    main()