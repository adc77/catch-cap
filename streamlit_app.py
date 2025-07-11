import streamlit as st
import requests
import json

# Configure page
st.set_page_config(
    page_title="Detect HaLLuCinations",
    page_icon="🤖",
    layout="wide"
)

# API base URL
API_BASE_URL = "http://localhost:7000"

def call_generate_api(query):
    """Call the /generate endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={"query": query, "model": "gpt-4.1-nano", "temperature": 1.0},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling generate API: {str(e)}")
        return None

def call_detect_api(query):
    """Call the /detect/detailed endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/detect/detailed",
            json={"query": query},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling detection API: {str(e)}")
        return None

def display_detection_results(detection_data):
    """Display hallucination detection results"""
    if not detection_data or not detection_data.get("success"):
        st.error("Failed to analyze response for hallucinations")
        return
    
    st.subheader("Hallucination Analysis Results")
    
    # Main analysis result
    analysis_result = detection_data.get("analysis_result", "")
    confabulation_detected = detection_data.get("confabulation_detected", False)
    
    if confabulation_detected:
        st.error(f"Status: {analysis_result}")
    else:
        st.success(f"Status: {analysis_result}")
    
    # Create two columns for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Initial Model Response")
        model_responses = detection_data.get("model_responses", [])
        if model_responses:
            for i, response in enumerate(model_responses, 1):
                st.text_area(f"Response {i}:", value=response, height=100, disabled=True)
    
    with col2:
        st.subheader("Web Search Result (Verified Answer)")
        web_answer = detection_data.get("web_answer", "No web answer available")
        st.text_area("Verified Answer:", value=web_answer, height=200, disabled=True)

    st.subheader("Reasoning for Hallucination/Confabulation")
    reasoning = detection_data.get("reasoning", "No reasoning available")
    st.text_area("Reasoning:", value=reasoning, height=100, disabled=True)
    
    # Additional details
    st.subheader("Analysis Details")
    comparison_result = detection_data.get("comparison_result", "No comparison available")
    st.text_area("Comparison Analysis:", value=comparison_result, height=100, disabled=True)
    
    # Technical details in expander
    with st.expander("Technical Details"):
        entropy_score = detection_data.get("entropy_score", 0)
        entropy_threshold = detection_data.get("entropy_threshold", 0.3)
        is_confident = detection_data.get("is_confident", False)
        
        st.write(f"Entropy Score: {entropy_score:.6f}")
        st.write(f"Entropy Threshold: {entropy_threshold}")
        st.write(f"Model Confidence: {'High' if is_confident else 'Low'}")
        
        web_search_summary = detection_data.get("web_search_summary", {})
        if web_search_summary:
            st.write("Web Search Summary:")
            st.json(web_search_summary)

# Main app
def main():
    st.title("Detect HaLLuCinations")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "current_response" not in st.session_state:
        st.session_state.current_response = ""
    if "show_detection" not in st.session_state:
        st.session_state.show_detection = False
    
    # Chat input
    user_input = st.text_input("Enter your question:", placeholder="Type your question here...")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        generate_clicked = st.button("Generate Response", type="primary")
    
    if generate_clicked and user_input.strip():
        st.session_state.current_query = user_input
        st.session_state.show_detection = False
        
        with st.spinner("Generating response..."):
            response_data = call_generate_api(user_input)
            
            if response_data and response_data.get("success"):
                st.session_state.current_response = response_data.get("response", "")
                
                # Add to messages
                st.session_state.messages.append({
                    "user": user_input,
                    "assistant": st.session_state.current_response,
                    "tokens": response_data.get("tokens_used", {}),
                    # "cost": response_data.get("cost", 0)
                })
            else:
                st.error("Failed to generate response")
    
    # Display current conversation
    if st.session_state.current_query and st.session_state.current_response:
        # st.subheader("Current Conversation")
        
        # User message
        st.write("**You:**")
        st.write(st.session_state.current_query)
        
        # Assistant response
        st.write("**Model Response:**")
        st.write(st.session_state.current_response)
        
        # Token usage info
        if st.session_state.messages:
            latest_message = st.session_state.messages[-1]
            tokens_used = latest_message.get("tokens", {})
            # cost = latest_message.get("cost", 0)
            
            if tokens_used:
                st.caption(f"Tokens used: {tokens_used.get('total_tokens', 0)}")
        
        # Hallucination detection button
        st.markdown("---")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            detect_clicked = st.button("Check for Hallucination/Confabulation?", type="primary")
        
        if detect_clicked:
            st.session_state.show_detection = True
            
            with st.spinner("Analyzing response for hallucinations/Confabulation..."):
                detection_data = call_detect_api(st.session_state.current_query)
                
                if detection_data:
                    display_detection_results(detection_data)
                else:
                    st.error("Failed to analyze response")
        
        elif st.session_state.show_detection:
            # Re-run detection to show results
            with st.spinner("Analyzing response for hallucinations/Confabulation..."):
                detection_data = call_detect_api(st.session_state.current_query)
                
                if detection_data:
                    display_detection_results(detection_data)
    
    # Conversation history
    if len(st.session_state.messages) > 1:
        st.markdown("---")
        st.subheader("Conversation History")
        
        for i, message in enumerate(reversed(st.session_state.messages[:-1]), 1):
            with st.expander(f"Conversation {len(st.session_state.messages) - i}"):
                st.write("**You:**")
                st.write(message["user"])
                st.write("**Model Response:**")
                st.write(message["assistant"])
                
                tokens = message.get("tokens", {})
                # cost = message.get("cost", 0)
                if tokens:
                    st.caption(f"Tokens: {tokens.get('total_tokens', 0)}")
    
    # Clear conversation button
    if st.session_state.messages:
        st.markdown("---")
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.current_query = ""
            st.session_state.current_response = ""
            st.session_state.show_detection = False
            st.rerun()

if __name__ == "__main__":
    main()