import streamlit as st
from pymongo import MongoClient
from datetime import datetime, timedelta
import json
from O365 import Account
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from langchain.llms import Ollama
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pydantic models for function parameters
class MeetingDetails(BaseModel):
    client_name: str = Field(..., description="Name of the client to schedule meeting with")
    meeting_date: str = Field(..., description="Date of the meeting in YYYY-MM-DD format")
    meeting_time: str = Field(..., description="Time of the meeting in HH:MM format")
    duration: int = Field(..., description="Duration of meeting in minutes")

class EmailDetails(BaseModel):
    recipient_email: str = Field(..., description="Email address of the recipient")
    subject: str = Field(..., description="Subject of the email")
    body: str = Field(..., description="Body content of the email")

class CalendarEventDetails(BaseModel):
    start_time: str = Field(..., description="Start time of the event in ISO format")
    end_time: str = Field(..., description="End time of the event in ISO format")
    attendees: List[str] = Field(..., description="List of attendee email addresses")
    subject: str = Field(..., description="Subject of the calendar event")

# Custom tools for function calling
class DatabaseTool(BaseTool):
    name = "query_client_database"
    description = "Queries the MongoDB database to find client information"
    args_schema = MeetingDetails

    def _run(self, client_name: str, **kwargs) -> Dict:
        client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        db = client['meeting_scheduler']
        collection = db['clients']
        
        result = collection.find_one({"name": client_name})
        if result:
            return {"found": True, "email": result["email"]}
        return {"found": False, "email": None}

class EmailTool(BaseTool):
    name = "send_email"
    description = "Sends email to specified recipients"
    args_schema = EmailDetails

    def _run(self, recipient_email: str, subject: str, body: str) -> bool:
        msg = MIMEMultipart()
        msg['From'] = os.getenv('EMAIL_SENDER')
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(os.getenv('EMAIL_SENDER'), os.getenv('EMAIL_PASSWORD'))
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            return False

class CalendarTool(BaseTool):
    name = "create_calendar_event"
    description = "Creates an event in Outlook calendar"
    args_schema = CalendarEventDetails

    def _run(self, start_time: str, end_time: str, attendees: List[str], subject: str) -> bool:
        try:
            account = Account((os.getenv('OUTLOOK_CLIENT_ID'), os.getenv('OUTLOOK_CLIENT_SECRET')))
            if account.authenticate():
                schedule = account.schedule()
                calendar = schedule.get_default_calendar()
                
                event = calendar.new_event()
                event.subject = subject
                event.start = datetime.fromisoformat(start_time)
                event.end = datetime.fromisoformat(end_time)
                event.attendees.add(attendees)
                event.save()
                return True
            return False
        except Exception as e:
            return False

# Function to create the Gemma model with function calling
def create_llm():
    tools = [DatabaseTool(), EmailTool(), CalendarTool()]
    
    function_descriptions = []
    for tool in tools:
        function_descriptions.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.args_schema.schema()["properties"],
                "required": list(tool.args_schema.schema()["properties"].keys())
            }
        })
    
    system_prompt = f"""You are a meeting scheduling assistant that helps process natural language requests into structured function calls.
Available functions: {json.dumps(function_descriptions, indent=2)}
Process the user's request and call the appropriate functions in the correct order."""

    return Ollama(
        model="gemma",
        callback_manager=CallbackManager(),
        system=system_prompt
    )

# Streamlit app
def main():
    st.title("AI Meeting Scheduler with Function Calling")
    
    user_prompt = st.text_input(
        "Enter your meeting request:",
        "Schedule a meeting with John Doe tomorrow at 2 PM for 1 hour"
    )
    user_email = st.text_input("Enter your email:")
    
    if st.button("Schedule Meeting"):
        if not user_email:
            st.error("Please enter your email address")
            return

        llm = create_llm()
        
        try:
            # First function call to extract meeting details
            response = llm.predict(f"""
            Process this meeting request and extract the client name to query the database:
            {user_prompt}
            """)
            
            # Query database
            db_tool = DatabaseTool()
            client_info = db_tool._run(json.loads(response)["client_name"])
            
            if not client_info["found"]:
                st.error("Client not found in database")
                return
                
            # Extract complete meeting details
            meeting_details_response = llm.predict(f"""
            Extract complete meeting details from this request to create calendar event:
            {user_prompt}
            Include the following attendees: {user_email}, {client_info['email']}
            """)
            
            meeting_details = json.loads(meeting_details_response)
            
            # Create calendar event
            calendar_tool = CalendarTool()
            calendar_success = calendar_tool._run(
                start_time=f"{meeting_details['meeting_date']}T{meeting_details['meeting_time']}",
                end_time=f"{meeting_details['meeting_date']}T{meeting_details['meeting_time']}",
                attendees=[user_email, client_info['email']],
                subject="Business Meeting"
            )
            
            if calendar_success:
                st.success("Calendar event created successfully")
                
                # Send emails
                email_tool = EmailTool()
                email_body = f"""
                Meeting Details:
                Date: {meeting_details['meeting_date']}
                Time: {meeting_details['meeting_time']}
                Duration: {meeting_details['duration']} minutes
                """
                
                email_success = email_tool._run(
                    recipient_email=client_info['email'],
                    subject="Meeting Invitation",
                    body=email_body
                )
                
                if email_success:
                    st.success("Meeting invitation sent successfully")
                else:
                    st.error("Failed to send email invitation")
            else:
                st.error("Failed to create calendar event")
                
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()