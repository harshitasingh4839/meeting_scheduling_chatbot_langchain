import streamlit as st
from pymongo import MongoClient
import requests
import json
import re
import spacy
import asyncio
from fuzzywuzzy import fuzz
from requests.exceptions import HTTPError, ConnectionError
from typing import Dict, Optional, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self, uri: str = 'mongodb://localhost:27017/'):
        self.uri = uri
        self.client: Optional[MongoClient] = None
        self.db = None

    def connect(self) -> Optional[Any]:
        """Establish connection to MongoDB with proper error handling."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.db = self.client['meeting_scheduling']
            # Verify connection
            self.client.server_info()
            return self.db
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
            st.error(f"Database connection error: {str(e)}")
            return None

    def close(self):
        """Safely close the database connection."""
        if self.client:
            self.client.close()

class ClientManager:
    def __init__(self, db):
        self.db = db
        self.collection = db['clients']

    def check_client(self, client_name: str) -> Dict[str, Any]:
        """
        Check if client exists in database using fuzzy matching.
        Returns a dictionary with client existence status and email.
        """
        try:
            if not client_name:
                return {'exists': False, 'email': None}

            clients = list(self.collection.find())  # Convert cursor to list
            if not clients:
                return {'exists': False, 'email': None}

            best_match = max(
                ((client, fuzz.ratio(client['name'].lower(), client_name.lower()))
                 for client in clients),
                key=lambda x: x[1],
                default=(None, 0)
            )

            if best_match[0] and best_match[1] > 90:
                return {'exists': True, 'email': best_match[0]['email']}
            return {'exists': False, 'email': None}

        except Exception as e:
            logger.error(f"Error querying MongoDB: {str(e)}")
            return {'exists': False, 'email': None}

class NameExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("Please install the English language model: python -m spacy download en_core_web_sm")
            raise

    def extract_person_name(self, text: str) -> Dict[str, str]:
        """Extract person name from text using NER."""
        try:
            doc = self.nlp(text)
            
            # Look for PERSON entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Validate name format
                    name = ent.text.strip()
                    if re.match(r"^[a-zA-Z\s'-]+$", name):
                        return {"client_name": name}
            
            return {"client_name": ""}
            
        except Exception as e:
            logger.error(f"Name extraction error: {str(e)}")
            return {"client_name": ""}

class OllamaClient:
    def __init__(self, base_url: str = 'http://localhost:11434'):
        self.base_url = base_url
        self.available_models = self.get_available_models()

    def get_available_models(self) -> list:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f'{self.base_url}/api/tags')
            if response.status_code == 200:
                return [model['name'] for model in response.json()['models']]
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def check_ollama_status(self) -> bool:
        """Check if Ollama server is running and responsive."""
        try:
            response = requests.get(f'{self.base_url}/api/version')
            return response.status_code == 200
        except:
            return False

    def generate_response(self, prompt: str, client_info: Dict[str, Any]) -> str:
        """Generate response using Ollama API with retries and fallback."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        # Prepare the message
        system_prompt = (
            f"You are a helpful meeting scheduling assistant. "
            f"{'The person mentioned is a registered client with email: ' + client_info['email'] if client_info['exists'] else 'The person mentioned is not found in our client database.'} "
            "Please provide a professional and courteous response regarding their meeting request."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Try different models in order of preference
        models_to_try = ['gemma:2b', 'llama2', 'mistral']
        available_models = set(self.available_models)

        for model in models_to_try:
            if model not in available_models:
                continue

            for attempt in range(max_retries):
                try:
                    data = {
                        "model": model,
                        "messages": messages,
                        "stream": False
                    }

                    response = requests.post(
                        f'{self.base_url}/api/chat',
                        json=data,
                        timeout=30  # 30 seconds timeout
                    )

                    if response.status_code == 200:
                        return response.json()['message']['content']

                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout with model {model}, attempt {attempt + 1}")
                except Exception as e:
                    logger.error(f"Error with model {model}, attempt {attempt + 1}: {str(e)}")

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        # Fallback response if all models fail
        return self.generate_fallback_response(client_info)

    def generate_fallback_response(self, client_info: Dict[str, Any]) -> str:
        """Generate a fallback response when LLM is unavailable."""
        if client_info['exists']:
            return (
                f"I see that you're a registered client with email: {client_info['email']}. "
                "I apologize, but I'm currently experiencing some technical difficulties. "
                "Please try again later or contact our support team directly for immediate assistance."
            )
        else:
            return (
                "I notice you're not in our client database. "
                "While I'm experiencing some technical difficulties, "
                "you can register as a new client through our website or contact our support team for assistance."
            )

def main():
    st.set_page_config(page_title="Meeting Scheduling Chatbot", layout="wide")
    st.title("Meeting Scheduling Chatbot")

    # Initialize components
    db_connection = DatabaseConnection()
    db = db_connection.connect()

    if db is None:
        st.error("Failed to connect to database. Please check if MongoDB is running.")
        return

    try:
        client_manager = ClientManager(db)
        name_extractor = NameExtractor()
        ollama_client = OllamaClient()

        # Check Ollama status
        if not ollama_client.check_ollama_status():
            st.warning("⚠️ Ollama service is not running. Please start Ollama service first.")
            st.info("To start Ollama, open a terminal and run: `ollama serve`")
            return

        # Display available models
        available_models = ollama_client.available_models
        if available_models:
            st.success(f"✓ Ollama is running with available models: {', '.join(available_models)}")
        else:
            st.warning("⚠️ No models available. Please pull a model using: `ollama pull gemma:2b`")
            return

        st.write("Enter your prompt with a client's name to get information")
        user_prompt = st.text_area("Enter your prompt:", key="user_prompt")

        if st.button("Get Response"):
            if not user_prompt:
                st.warning("Please enter a prompt")
                return

            with st.spinner("Processing..."):
                # Extract client name
                extracted_name = name_extractor.extract_person_name(user_prompt)
                client_name = extracted_name.get('client_name')

                if not client_name:
                    st.error("Could not extract a valid person name from the prompt")
                    return

                # Check client in database
                client_info = client_manager.check_client(client_name)

                # Get response
                response = ollama_client.generate_response(user_prompt, client_info)

                # Display results
                st.subheader("Response:")
                st.write(response)

                st.subheader("Client Status:")
                if client_info['exists']:
                    st.success(f"✓ {client_name} is a registered client")
                    st.info(f"📧 Email: {client_info['email']}")
                else:
                    st.warning(f"⚠️ {client_name} is not found in our client database")

    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        db_connection.close()

if __name__ == "__main__":
    main()