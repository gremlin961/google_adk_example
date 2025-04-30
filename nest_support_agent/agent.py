# Copyright 2024 Google, LLC. This software is provided as-is,
# without warranty or representation for any use or purpose. Your
# use of it is subject to your agreement with Google.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example Agent Workflow using Google's ADK
# 
# This notebook provides an example of building an agentic workflow with Google's new ADK. 
# For more information please visit  https://google.github.io/adk-docs/



# Vertex AI Modules
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part, Tool, ChatSession, FunctionDeclaration, grounding, GenerationResponse
from vertexai.preview import rag # Import the RAG (Retrieval-Augmented Generation) module

# Vertex Agent Modules
from google.adk.agents import Agent # Base class for creating agents
from google.adk.runners import Runner # Class to run agent interactions
from google.adk.sessions import InMemorySessionService # Simple session management (non-persistent)
from google.adk.artifacts import InMemoryArtifactService, GcsArtifactService # In-memory artifact storage
from google.adk.tools.agent_tool import AgentTool # Wrapper to use one agent as a tool for another
from google.adk.tools import ToolContext

# Vertex GenAI Modules (Alternative/Legacy way to interact with Gemini, used here for types)
import google.genai
from google.genai import types as types # Used for structuring messages (Content, Part)

# Google Cloud AI Platform Modules
from google.cloud import aiplatform_v1beta1 as aiplatform # Specific client for RAG management features
from google.cloud import storage # Client library for Google Cloud Storage (GCS)

# Other Python Modules
#import base64 # Not used in the final script
#from IPython.display import Markdown # Not used in the final script
import asyncio # For running asynchronous agent interactions
import requests # For making HTTP requests (to the mock ticket server)
import os # For interacting with the operating system (paths, environment variables)
from typing import List, Dict, TypedDict, Any # For type hinting
import json # For working with JSON data (API requests/responses)
from urllib.parse import urlparse # For parsing GCS bucket URIs
import warnings # For suppressing warnings
import logging # For controlling logging output
import mimetypes # For detecting mime types of files
import io

# Ignore all warnings
warnings.filterwarnings("ignore")
# Set logging level to ERROR to suppress informational messages
logging.basicConfig(level=logging.ERROR)

# --- Configuration ---
project_id = "rkiles-demo-host-vpc" # Your GCP Project ID
location = "global" # Vertex AI RAG location (can be global for certain setups)
region = "us-central1" # Your GCP region for Vertex AI resources and GCS bucket

corpa_name = "nest-rag-corpus" # Display name for the Vertex AI RAG Corpus

ticket_server_url = "http://localhost:8001" # URL of the mock ticketing system web service

# --- Environment Setup ---
# Set environment variables required by some Google Cloud libraries
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1" # Instructs the google.genai library to use Vertex AI backend
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = region

# --- Initialize Vertex AI SDK ---
# Initialize the Vertex AI client library with project and location/region details
vertexai.init(project=project_id, location=region)







# --- Agent Tool Definitions ---
# @title Define Tools for creating a ticket, adding notes to a ticket, add a file to the session, and getting the GCS URI

# Tool function to create a support ticket via a mock API
def create_ticket(ticket_data: Dict) -> Dict:
    """
    Creates a ticket via the /ticket endpoint of the mock ticket server.

    Args:
        ticket_data: A dictionary containing ticket details like
                     { "ticket_id": ..., "description": ..., "customer_id": ..., "contact_name": ... }.

    Returns:
        A dictionary containing the API response (usually success status or created ticket details),
        or an error dictionary if the request fails.
    """
    url = f"{ticket_server_url}/ticket" # Construct the API endpoint URL
    headers = {"Content-Type": "application/json"} # Set request headers

    try:
        # Send a POST request with the ticket data as JSON
        response = requests.post(url, headers=headers, data=json.dumps(ticket_data))
        # Raise an exception for bad HTTP status codes (4xx or 5xx)
        response.raise_for_status()
        # Return the parsed JSON response from the API
        return response.json()
    except requests.exceptions.RequestException as e:
        # Catch potential request errors (network issues, invalid URL, etc.)
        print(f"Error in create_ticket: {e}") # Log the error
        return {"error": f"Request failed: {e}"} # Return an error dictionary

# Tool function to add a note to an existing ticket via a mock API
def add_note(ticket_id: int, contact_name: str, note: str) -> Dict[str, Any]:
    """
    Adds a note to a specific ticket via the /notes endpoint of the mock ticket server.

    Args:
        ticket_id (int): The ID of the ticket to add the note to.
        contact_name (str): The name of the person adding the note.
        note (str): The content of the note.

    Returns:
        Dict[str, Any]: A dictionary containing the API response (e.g., success confirmation),
                       or an error dictionary if the request fails.
    """
    url = f"{ticket_server_url}/notes" # Construct the API endpoint URL
    headers = {"Content-Type": "application/json"} # Set request headers

    # Construct the data payload for the API request
    note_data = {
        "ticket_id": ticket_id,
        "contact_name": contact_name,
        "note": note
    }

    try:
        # Send a POST request with the note data as JSON
        response = requests.post(url, headers=headers, data=json.dumps(note_data))
        # Raise an exception for bad HTTP status codes
        response.raise_for_status()
        # Return the parsed JSON response from the API
        return response.json()
    except requests.exceptions.RequestException as e:
        # Catch potential request errors
        print(f"Error in add_note: {e}") # Log the error
        return {"error": f"Request failed: {e}"} # Return an error dictionary

# Tool function to represent adding a file to the agent's context (for reasoning)
# Note: The implementation using types.Part.from_uri might have issues or specific requirements
# within the ADK framework, hence the return type change to `str` in the user's code.
# The core idea is to signal *which* file should be considered by the next agent.
# def add_file_to_session(uri: str) -> types.Content: # Original intended signature
def add_file_to_session(uri: str, tool_context: ToolContext) -> str: # Modified signature as per user code
    """
    Adds a specific file from Google Cloud Storage (GCS) to the current session state for agent processing.

    This function takes a GCS URI for a file and wraps it in a `types.Content` object.
    This object is then typically used to make the file's content accessible to the
    agent for tasks like summarization, question answering, or data extraction
    related specifically to that file within the ongoing conversation or session.
    The MIME type is assumed to be inferred by the underlying system or defaults.

    Use this function *after* you have identified a specific GCS URI (e.g., using
    `get_gcs_uri` or similar) that you need the agent to analyze or reference directly.

    Args:
        uri: str - The complete Google Cloud Storage URI of the file to add.
                 Must be in the format "gs://bucket_name/path/to/file.pdf".
                 Example: "gs://my-doc-bucket/reports/q1_report.pdf"

    Returns:
         types.Content - A structured Content object representing the referenced file.
                       This object has `role='user'` and contains a `types.Part`
                       that holds the reference to the provided GCS URI.
                       This Content object can be passed to the agent in subsequent calls.
    """
    
    # Determine the bucket name and blob names
    path_part = uri[len("gs://"):]
    # Split only on the first '/' to separate bucket from the rest
    bucket_name, blob_name = path_part.split('/', 1)

    # Initialize GCS client
    storage_client = storage.Client()

    # Get the bucket and blob objects
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the file content as bytes
    file_bytes = blob.download_as_bytes()

    # Determine the mime type of the file based on the file extension
    # Split the URI by the last '/'
    parts = uri.rsplit('/', 1)
    filename = parts[-1]

    # Detect the MIME type of the file
    mime_type, encoding = mimetypes.guess_type(filename)
    #print(f"Detected MIME type: {mime_type}")

    # This part attempts to create a Content object referencing the GCS URI.
    file_artifact = types.Part(
        inline_data=types.Blob(
            data=file_bytes,
            mime_type=mime_type
        )
    )
    
    version = tool_context.save_artifact(
        filename=filename, artifact=file_artifact
    )
    #content = types.Content(
    #    role='user',
    #    parts=[
    #        types.Part.from_data(
    #            data=file_bytes, 
    #            mime_type=mime_type
    #        )
    #    ]
    #)
  
   
    # Add file to Session instead of Artifacts
    #content = types.Content(
    #    role='user',
    #    parts=[
    #        # Try passing ONLY the uri positionally, based on the error message "takes 1 positional argument"
    #        types.Part.from_uri(
    #            file_uri=uri,
    #            mime_type="application/pdf",
    #        )
    #    ]
    #)

    return f'Artifact {filename} has been created with version {version}'


# Tool function to query the RAG corpus and retrieve relevant GCS URIs
def get_gcs_uri(query: str) -> str:
    """
    Queries the configured Vertex AI RAG corpus to find relevant document URIs based on the input query.

    Args:
        query (str): The natural language query to search for in the RAG corpus.

    Returns:
        str: A JSON string representing a list of unique GCS URIs of the relevant documents found.
             Returns an empty JSON list '[]' if no relevant documents are found.
             Example: '["gs://my-bucket/doc1.pdf", "gs://my-bucket/report_q3.txt"]'
    """
    if not rag_corpus: # Check if rag_corpus object is available
        return json.dumps({"error": "RAG Corpus not initialized"})

    print(f"Tool 'get_gcs_uri' called with query: {query}")
    try:
        # Perform a retrieval query against the RAG corpus
        query_response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name, # Use the name (resource ID) of the corpus
                    # Optional: Filter by specific file IDs if needed
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            text=query, # The user's query
            similarity_top_k=10,  # Optional: Max number of results to retrieve
            vector_distance_threshold=0.5,  # Optional: Similarity threshold (lower means more similar)
        )

        # Extract the source URIs from the response contexts
        uri_set = set() # Use a set to automatically handle duplicates
        if query_response.contexts and hasattr(query_response.contexts, 'contexts'):
             for context in query_response.contexts.contexts:
                if hasattr(context, 'source_uri'):
                    uri_set.add(context.source_uri)

        # Convert the set of unique URIs to a list and then to a JSON string
        doc_uris_json = json.dumps(list(uri_set))
        print(f"Tool 'get_gcs_uri' found URIs: {doc_uris_json}")
        return doc_uris_json
    except Exception as e:
        print(f"Error during RAG query in get_gcs_uri: {e}")
        return json.dumps({"error": f"RAG query failed: {e}"})


# --- RAG Corpus Initialization ---
# List existing RAG corpora in the project/location
print("Checking for existing RAG Corpora...")
existing_corpora = rag.list_corpora()

# Print the raw response for debugging if needed
# print(existing_corpora)

# Variable to hold the target RAG corpus object
rag_corpus = None # Initialize to None

# Iterate through the listed RAG corpora
# Note: The actual corpora are usually in an attribute like 'rag_corpora' of the response object
corpora_list = getattr(existing_corpora, 'rag_corpora', []) # Safely get the list
for corpus in corpora_list:
    # Check if the corpus has a 'display_name' attribute and if it matches the desired name
    if getattr(corpus, 'display_name', None) == corpa_name:
        print(f"Existing RAG Corpus found with display name '{corpa_name}'. Using: {corpus.name}")
        rag_corpus = corpus # Assign the found corpus object
        print(f"This corpus ('{corpus.name}') contains the following files:")
        try:
            # List the files within the found corpus
            files_pager = rag.list_files(corpus.name) # Use the corpus resource name
            for file in files_pager:
                # Print the display name of each file, handling potential missing attribute
                print(f" - {getattr(file, 'display_name', 'N/A (No display name)')}")
        except Exception as e:
            # Handle errors during file listing (e.g., permissions)
            print(f"Warning: Could not list files for corpus {corpus.name}. Error: {e}")
        break # Exit the loop once the matching corpus is found

# If the loop finished without finding the corpus
if rag_corpus is None:
    print(f"No existing RAG corpus found with display name '{corpa_name}'. You will need to create this RAG corpus to continue.")
    exit



# --- Sub-Agent Definitions ---
# @title Define RAG, Reasoning and Notes Sub-Agents

# --- RAG Agent ---
# This agent's role is to use the RAG tool (get_gcs_uri) to find relevant document URIs.
rag_agent = None
rag_agent = Agent(
    model="gemini-2.0-flash-001", # Specifies the LLM to power this agent
    name="rag_agent",             # Unique name for this agent
    instruction=                  # Prompt defining the agent's behavior and goal
    """
        You are a customer support agent specialized in document retrieval.
        Your sole purpose is to identify the Google Cloud Storage (GCS) URIs of support documents relevant to the user's query.
        Use the 'get_gcs_uri' tool to find the closest matching document(s).
        The tool will return a JSON string containing a list of URIs, like '["gs://bucket/file1.pdf", "gs://bucket/file2.txt"]'.
        Return only this JSON string of URIs back to the main agent. Do not attempt to answer the user's question directly or summarize the documents.
    """,
    description="Retrieves relevant document GCS URIs from the RAG system based on a query.", # Description used when this agent is a tool for another agent
    tools=[
        get_gcs_uri # Make the get_gcs_uri function available as a tool to this agent
    ],
)


# -- Reasoning Agent ---
# This agent's role is to generate troubleshooting steps based on document content
# (which is expected to be in the context after add_file_to_session is called).
reasoning_agent = None
reasoning_agent = Agent(
    model="gemini-2.5-pro-exp-03-25", # Advanced model for complex tasks and reasoning
    #model="gemini-2.5-flash-preview-04-17", # Alternate Model that can be used to help address resource constraints with more advanced models
    name="reasoning_agent",
    instruction=
    """
        You are a technical support specialist. Your task is to create a clear, step-by-step troubleshooting plan based on the provided support documents (which are now in your context).
        The user's original problem description will be provided.
        Analyze the information within the document(s) made available in this session context.
        Clearly state which document(s) contain the information used for the plan.
        The plan should outline the actions a Nest technical support representative should take to resolve the customer's issue.
        Format the output as a numbered list of steps.
        Ensure the output contains only plain text suitable for adding to a support ticket note (no markdown, formatting, etc.).

        Example Output:
        Based on the document 'gs://bucket/nest_install_guide.pdf':
        Step 1: Verify thermostat wiring matches the guide's diagram.
        Step 2: Check for power delivery to the thermostat base.
        Step 3: Follow the pairing instructions in section 4.
    """,
    description="Generates a troubleshooting plan based on information from provided documents.",
    tools=[
        # No specific tools needed here; relies on context provided by the root agent.
        # add_file_to_session, # This tool is called by the ROOT agent *before* calling this reasoning agent.
    ],
)


# --- Notes Agent ---
# This agent's role is to format the troubleshooting plan and add it as a note to the ticket system.
notes_agent = None
notes_agent = Agent(
    model="gemini-2.0-flash-001",
    #model="gemini-2.0-flash-exp", # Model that supports Audio input and output
    name="notes_agent",
    instruction=
    """
        You are a customer support assistant agent. Your primary task is to generate informative and well-structured notes for customer support tickets. 
        These notes will be added to the ticket's history and used by customer support representatives to understand the issue, track progress, and provide consistent service.
        Step 1: Check if you have the user's ticket ID and contact name (you can use their provided name or email as contact name). If not, politely ask for the informaiton you are missing.
        Step 2: Once you have the ticket ID, contact name, and the troubleshooting steps, use the 'add_note' tool to add the notes to the ticket.
        Step 3: Confirm to the user that the ticket has been updated with the troubleshooting plan.
        Step 4: Thank the customer for their time and to have a nice day.
    """,
    description="Adds the generated troubleshooting plan as a note to the specified support ticket.",
    tools=[
        add_note, # Make the add_note function available as a tool
    ],
)


# --- Root Agent Definition ---
# @title Define the Root Agent with Sub-Agents

# Initialize root agent variables
root_agent = None
runner_root = None # Initialize runner variable (although runner is created later)

    # Define the root agent (coordinator)
nest_agent_team = Agent(
    name="nest_support_agent",    # Name for the root agent
    model="gemini-2.0-flash-001", # Model for the root agent (orchestration)
    #model="gemini-2.0-flash-exp", # Model that supports Audio input and output 
    description="The main coordinator agent. Handles user requests and delegates tasks to specialist sub-agents and tools.", # Description (useful if this agent were itself a sub-agent)
    instruction=                  # The core instructions defining the workflow
    """
        You are the lead Nest customer support coordinator agent. Your goal is to understand the customer's issue, find relevant documentation, generate a troubleshooting plan, and log the plan into their support ticket.

        You have access to specialized tools and sub-agents:
        1. Tool `add_file_to_session`: Use this tool *after* getting GCS URIs. Provide ONE GCS URI (e.g., "gs://bucket/doc.pdf") to this tool. The tool prepares the file content for context. Call this tool for EACH relevant URI returned by the rag_agent.
        2. Sub-Agent `rag_agent`: Call this agent first with the user's problem description to get a JSON list of relevant GCS document URIs.
        3. Sub-Agent `reasoning_agent`: Call this agent *after* using `add_file_to_session` for all relevant URIs. Provide the user's problem and indicate that the relevant documents are now in context. This agent will return the troubleshooting steps.
        4. Sub-Agent `notes_agent`: Call this agent last. You need the `ticket_id` (integer), `contact_name` (string, use your name "Nest Support Agent"), and the `note` (string, the troubleshooting steps from reasoning_agent). Ask the user for their ticket ID and name/email if you don't have it.

        Start by greeting the user and ask no more than 1-2 questions to better understand the Nest product they are using and their issue.
        IMPORTANT - When you make a tool call or hand off to another agent, politely ask the user to please wait while you research the issue.
        Whne calling the `rag_agent` provide the user's issue description. Extract the GCS URIs from the JSON list it returns.
        If URIs are returned from the 'rag_agent':
            - For EACH URI in the list, call the `add_file_to_session` tool with that single URI. 
        After processing all relevant files, call the `reasoning_agent`. Provide the user's original problem description and explicitly state that the necessary documents are in the context. Capture the troubleshooting steps it returns. 
    """,
    tools=[
        add_file_to_session,      # Make the file session tool directly available to the root agent
        AgentTool(agent=rag_agent), # Make the rag_agent available as a tool
        AgentTool(agent=reasoning_agent) # Make the reasoning_agent available as a tool
        # Note: notes_agent is listed as a sub_agent, not a direct tool here.
        # This implies delegation rather than direct tool calling for notes_agent.
    ],
        # List agents that this agent can delegate tasks to.
        # The root agent decides *when* to invoke these based on its instructions.
    sub_agents=[notes_agent]
)

# Assign the created agent to the root_agent variable for clarity in the next step
root_agent = nest_agent_team




