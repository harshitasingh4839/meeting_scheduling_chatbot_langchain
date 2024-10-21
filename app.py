import streamlit as st
from pymongo import MongoClient
import requests
import json
import re

# MongoDB connection
def connect_to_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['meeting_scheduling']
    return db

# Function to check if client exists in database
def check_client_in_db(db, client_name):
    client_collection = db['clients']
    client = client_collection.find_one({'name': client_name})
    if client:
        return {'exists': True, 'email': client['email']}
    return {'exists': False, 'email': None}


import spacy

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")


# Function to extract person name from text
def extract_person_name(text):
    """
    Extract the client's name from the given text using Named Entity Recognition (NER).
    """
   
    function_definition = {
        "name": "extract_client_name",
        "description": "Extract the client's name from the given text",
        "parameters": {
            "type": "object",
            "properties": {
                "client_name": {
                    "type": "string",
                    "description": "The extracted client's name"
                }
            },
            "required": ["client_name"]
        }
    }

    doc = nlp(text)
    # Look for PERSON entities in the text
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return {"client_name": ent.text}
    return {"client_name": "Could not extract a person name"}
    
    
    # Prompt for name extraction
    system_prompt = """You are a helpful assistant that extracts person names from text. 
    Use the provided function to return the extracted name."""
    
    # Create the API request for Ollama
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    data = {
        "model": "gemma2:2b",
        "messages": messages,
        "functions": [function_definition],
        "function_call": {"name": "extract_person_name"}
    }
    
    # Make API call to local Ollama instance
    response = requests.post('http://localhost:11434/api/chat', json=data)
    
    if response.status_code == 200:
        try:
            function_call = json.loads(response.json()['message']['function_call'])
            return function_call['arguments']['person_name']
        except:
            return None
    return None

# Function to get LLM response
def get_llm_response(prompt, client_info):
    # Create system prompt based on client information
    if client_info['exists']:
        system_prompt = f"""You are a helpful assistant. The person mentioned is a client with email: {client_info['email']}.
        Include this information naturally in your response."""
    else:
        system_prompt = "You are a helpful assistant. The person mentioned is not found in our client database."
    
    # Create the API request for Ollama
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    data = {
        "model": "gemma:2b",
        "messages": messages
    }
    
    # Make API call to local Ollama instance
    response = requests.post('http://localhost:11434/api/chat', json=data)
    
    if response.status_code == 200:
        return response.json()['message']['content']
    return "Error: Could not get response from LLM"

# Streamlit UI
def main():
    st.title("Client Information Assistant")
    st.write("Enter your prompt with a client's name to get information")
    
    # Initialize MongoDB connection
    db = connect_to_mongodb()
    
    # User input
    user_prompt = st.text_area("Enter your prompt:")
    
    if st.button("Get Response"):
        if user_prompt:
            with st.spinner("Processing..."):
                # Extract client name
                extracted_name = extract_person_name(user_prompt)
                client_name = extracted_name.get('client_name')
                
                if client_name and client_name != "Could not extract a person name":
                    # Check client in database
                    client_info = check_client_in_db(db, client_name)
                    
                    # Get LLM response
                    response = get_llm_response(user_prompt, client_info)
                    
                    # Display response
                    st.write("Response:")
                    st.write(response)
                    
                    # Display client status
                    st.write("\nClient Status:")
                    if client_info['exists']:
                        st.success(f"{client_name} is a client")
                        st.info(f"Email: {client_info['email']}")
                    else:
                        st.warning(f"{client_name} is not found in our client database")
                else:
                    st.error("Could not extract a person name from the prompt")
        else:
            st.warning("Please enter a prompt")

if __name__ == "__main__":
    main()