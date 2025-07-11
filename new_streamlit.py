import streamlit as st
import requests
import json
from typing import Dict, Any

# Configure page
st.set_page_config(
    page_title="AI Chat with Confabulation Detection",
    page_icon="🤖",
    layout="wide"
)

# API base URL - make sure your FastAPI server is running on this port
API_BASE_URL = "http://localhost:7000"

def call_detect_detailed_api(query: str) -> Dict[str, Any]:
    """Call the /detect/detailed endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/detect/detailed",
            json={"query": query, "entropy_threshold": 0.3},
            timeout=120  # Longer timeout for detailed analysis
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling detection API: {str(e)}")
        return {}

def get_most_common_response(model_responses):
    """Get the most common response from multiple model responses"""
    if not model_responses:
        return "No response available"
    
    # For simplicity, return the first response
    # You could implement actual frequency analysis here if needed
    return model_responses[0]

def display_chat_message(role: str, content: str, is_confabulated: bool = False):
    """Display a chat message with appropriate styling"""
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            if is_confabulated:
                st.error("⚠️ Potential confabulation detected in this response!")
            st.write(content)

def display_confabulation_details(detection_data: Dict[str, Any]):
    """Display detailed confabulation analysis in an expandable section"""
    with st.expander("Detailed Confabulation Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correct Answer (from Web Search)")
            web_answer = detection_data.get("web_answer", "No web answer available")
            st.info(web_answer)
        
        with col2:
            st.subheader("Analysis Details")
            analysis_result = detection_data.get("analysis_result", "")
            entropy_score = detection_data.get("entropy_score", 0)
            st.write(f"**Status:** {analysis_result}")
            st.write(f"**Entropy Score:** {entropy_score:.4f}")
        
        st.subheader("Reasoning for Confabulation Detection")
        reasoning = detection_data.get("reasoning", "No reasoning available")
        st.write(reasoning)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title
st.title("Detect HaLLuCinations")
st.caption("Chat and detect potential confabulations/haLLuCinations")

# Display chat history
for message in st.session_state.messages:
    display_chat_message(
        message["role"], 
        message["content"], 
        message.get("is_confabulated", False)
    )
    
    # Show confabulation details if available
    if message.get("detection_data") and message.get("is_confabulated"):
        display_confabulation_details(message["detection_data"])

# Chat input at the bottom
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_chat_message("user", prompt)
    
    # Process the query through confabulation detection
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question and generating response..."):
            detection_data = call_detect_detailed_api(prompt)
            
            if detection_data and detection_data.get("success"):
                # Get the most common response
                model_responses = detection_data.get("model_responses", [])
                response_content = get_most_common_response(model_responses)
                
                # Check for confabulation
                confabulation_detected = detection_data.get("confabulation_detected", False)
                analysis_result = detection_data.get("analysis_result", "")
                
                # Display response with confabulation warning if needed
                if confabulation_detected:
                    st.error("⚠️ Potential confabulation detected in this response!")
                
                st.write(response_content)
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "is_confabulated": confabulation_detected,
                    "detection_data": detection_data if confabulation_detected else None
                })
                
                # Show detailed analysis if confabulation detected
                if confabulation_detected:
                    display_confabulation_details(detection_data)
                
            else:
                error_message = "Sorry, I couldn't process your request. Please try again."
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "is_confabulated": False
                })

# Sidebar with instructions
with st.sidebar:
    st.header("How it works")
    st.write("""
    1. Ask any question in the chat
    2. If confabulation/hallucination is detected, you'll see a warning flag
       - The correct answer from web search
       - Detailed reasoning for the detection
    """)
    
    st.header("API Status")
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Not Available")
        st.caption("Make sure to run: `python confab_app.py`")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()