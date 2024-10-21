import spacy

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")

def extract_client_name(prompt):
    """
    Extract the client's name from the user's prompt using Named Entity Recognition (NER).
    """
    # Process the prompt using spaCy
    doc = nlp(prompt)
    
    # Look for PERSON entities in the text
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return {"client_name": ent.text}
    
    return {"client_name": "Could not extract a person name from the prompt"}

# Example prompt
prompt = "I want to schedule a meeting with John Doe on Wednesday from 6 pm"
client_info = extract_client_name(prompt)
print(client_info)
