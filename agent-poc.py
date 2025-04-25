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
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # In-memory artifact storage (not used explicitly here but part of ADK)
from google.adk.tools.agent_tool import AgentTool # Wrapper to use one agent as a tool for another

# Vertex GenAI Modules (Alternative/Legacy way to interact with Gemini, used here for types)
import google.genai
from google.genai import types # Used for structuring messages (Content, Part)

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

# Ignore all warnings
warnings.filterwarnings("ignore")
# Set logging level to ERROR to suppress informational messages
logging.basicConfig(level=logging.ERROR)

# --- Configuration ---
project_id = "<YOUR_PROJECT_ID>" # Your GCP Project ID
location = "global" # Vertex AI RAG location (can be global for certain setups)
region = "us-central1" # Your GCP region for Vertex AI resources and GCS bucket

corpa_name = "nest-rag-corpus" # Display name for the Vertex AI RAG Corpus

corpa_document_bucket = "gs://<YOUR_BUCKET>/nest/docs/" # Google Cloud Storage path where source documents for RAG are stored

local_documents = "./nest_docs/" # Local directory containing documents to upload to GCS

ticket_server_url = "http://ticket01:8000" # URL of the mock ticketing system web service

# --- Environment Setup ---
# Set environment variables required by some Google Cloud libraries
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1" # Instructs the google.genai library to use Vertex AI backend
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = region

# --- Initialize Vertex AI SDK ---
# Initialize the Vertex AI client library with project and location/region details
vertexai.init(project=project_id, location=region)

# --- GCS Bucket and Folder Setup ---
# Parse the GCS bucket URI to get the bucket name and prefix (folder path)
parsed_uri = urlparse(corpa_document_bucket)
bucket_name = parsed_uri.netloc
prefix = parsed_uri.path.lstrip('/')

# Create a GCS client
storage_client = storage.Client()

# Get the bucket object
bucket = storage_client.bucket(bucket_name)

# Check if the bucket exists, create it if it doesn't
if not bucket.exists():
    bucket.create()
    print(f"Bucket '{bucket_name}' created successfully.")
else:
    print(f"Bucket '{bucket_name}' already exists.")

# Ensure the specified folder path exists within the bucket.
# GCS doesn't have real folders, but prefixes. Creating an empty object
# simulates a folder structure for tools and browsing.
if prefix:
    # Ensure the prefix ends with '/' for folder simulation
    blob_name = f"{prefix}" if prefix.endswith('/') else f"{prefix}/"
    # Create a placeholder blob to represent the folder if it doesn't exist
    placeholder_blob = bucket.blob(blob_name + ".placeholder") # Use a placeholder file name
    if not placeholder_blob.exists():
        placeholder_blob.upload_from_string('') # Upload empty content
        print(f"Simulated folder '{corpa_document_bucket}' created.")
    else:
        print(f"Simulated folder '{corpa_document_bucket}' already exists.")

# --- Upload Local Documents to GCS ---
# Check if the specified local directory exists
if os.path.exists(local_documents) and os.path.isdir(local_documents):
    # Iterate over files in the local directory
    for filename in os.listdir(local_documents):
        local_file_path = os.path.join(local_documents, filename)
        # Check if it's actually a file (and not a subdirectory)
        if os.path.isfile(local_file_path):
            # Construct the destination path in GCS including the prefix
            gcs_blob_name = f"{prefix}{filename}"
            # Get the blob object for the destination
            blob = bucket.blob(gcs_blob_name)
            # Upload the local file to GCS
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded '{local_file_path}' to 'gs://{bucket_name}/{gcs_blob_name}'")
else:
    # Print a message if the local directory is not found
    print(f"Local directory '{local_documents}' does not exist or is not a directory.")

# --- Agent Interaction Function ---
# @title Define Agent Interaction Function
import asyncio
from google.genai import types # For creating message Content/Parts

# Define an asynchronous function to interact with an ADK agent runner
async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str): # <--- Added parameters for session context
    """Sends a query to the agent and prints the final response."""
    #print(f"\n>>> User Query: {query}")

    # Prepare the user's message in the ADK Content format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    # Default response text if the agent doesn't provide one
    final_response_text = "Agent did not produce a final response."

    # Key Concept: runner.run_async executes the agent logic and yields Events asynchronously.
    # We iterate through these events to capture the agent's actions and final response.
    # Use the passed-in user_id and session_id for maintaining conversation state.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution for debugging
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: event.is_final_response() indicates the agent's concluding message for this turn.
        if event.is_final_response():
            # Check if the event has content (usually the agent's text response)
            if event.content and event.content.parts:
                # Assume the text response is in the first part
                final_response_text = event.content.parts[0].text
            # Check if the agent escalated (e.g., encountered an error or needs human help)
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break # Stop processing events once the final response is found

    # Print the agent's final response for this turn
    print(f"Agent: {final_response_text}") # Added "Agent: " prefix

# --- RAG Corpus Management Functions ---

# Function to delete RAG corpora (requires aiplatform client)
# Note: This function definition exists but is not called in the main script flow.
def delete_rag_corpora(data: aiplatform.services.vertex_rag_data_service.pagers.ListRagCorporaPager):
    """Extracts 'name' values from a ListRagCorporaPager and attempts to delete them.

    Args:
        data: The ListRagCorporaPager object returned from listing RAG corpora.
    """
    names_list = []
    # First loop: Collect the names of all corpora from the pager object
    for rag_corpus in data:  # Iterate through the rag_corpus objects in the pager
        if hasattr(rag_corpus, 'name'):  # Check if the object has a 'name' attribute
             names_list.append(rag_corpus.name) # Add the name (resource ID) to the list

    # Second loop: Iterate through the collected names and delete each corpus
    for corpa_name in names_list:
        print(f"Attempting to delete RAG corpus: {corpa_name}")
        # Use the vertexai.preview.rag module to delete the corpus by its name
        # rag.get_corpus(name=corpa_name) # Getting is not strictly necessary before deleting
        try:
            rag.delete_corpus(name=corpa_name)
            print(f"Deleted RAG corpus: {corpa_name}")
        except Exception as e:
            print(f"Error deleting corpus {corpa_name}: {e}")

# Function to create a new RAG corpus and import files from GCS
def create_rag_corpora(display_name, source_bucket):
    """Creates a Vertex AI RAG Corpus and imports files from a GCS bucket.

    Args:
        display_name (str): The desired display name for the new RAG Corpus.
        source_bucket (str): The GCS URI (gs://...) pointing to the files to ingest.

    Returns:
        rag.RagCorpus: The created RagCorpus object.
    """
    # Specify the embedding model to use for the corpus
    EMBEDDING_MODEL = "publishers/google/models/text-embedding-004"
    embedding_model_config = rag.EmbeddingModelConfig(publisher_model=EMBEDDING_MODEL)

    print(f"Creating RAG Corpus with display name: {display_name}")
    # Create the RAG corpus using the vertexai.preview.rag module
    rag_corpus = rag.create_corpus(
        display_name=display_name,
        embedding_model_config=embedding_model_config
    )
    print(f"RAG Corpus created with name: {rag_corpus.name}")

    # Specify the GCS path containing the documents to import
    INPUT_GCS_BUCKET = source_bucket

    print(f"Importing files from {INPUT_GCS_BUCKET} into corpus {rag_corpus.name}...")
    # Import files from the specified GCS path into the created corpus
    # This process involves chunking the documents and generating embeddings.
    response = rag.import_files(
        corpus_name=rag_corpus.name,
        paths=[INPUT_GCS_BUCKET], # GCS paths must be in a list
        chunk_size=1024,  # Optional: Size of text chunks for processing
        chunk_overlap=100,  # Optional: Overlap between chunks
        max_embedding_requests_per_min=900,  # Optional: Rate limiting for embedding generation
    )
    print(f"File import process started. Response: {response}") # Note: Import is asynchronous

    # Example of uploading a single local file (commented out)
    # rag_file = rag.upload_file(
    #    corpus_name=rag_corpus.name,
    #    path="./test.txt",
    #    display_name="test.txt",
    #    description="my test file"
    # )

    # Return the created corpus object
    return rag_corpus

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
def add_file_to_session(uri: str) -> str: # Modified signature as per user code
    """
    Represents the action of making a file from GCS available to the agent's context for the current session.

    This function takes a GCS URI and is intended to signal that the content of this file
    should be accessible for subsequent processing steps within the agent's turn,
    particularly for the `reasoning_agent`. The actual mechanism of making the file content
    available might happen implicitly within the ADK framework when used correctly or might
    require specific handling not fully implemented here.

    Args:
        uri (str): The Google Cloud Storage URI of the file (e.g., "gs://bucket/file.pdf").

    Returns:
        str: Currently returns a string representation, likely indicating the file was processed.
             The original intent might have been to return a structured `types.Content` object.
    """
    print(f"Tool 'add_file_to_session' called with URI: {uri}")
    # This part attempts to create a Content object referencing the GCS URI.
    # The effectiveness depends on how the ADK runner/model handles this specific Part type.
    # Based on the change to return `str`, this Content object might not be directly used as intended.
    # content = types.Content(
    #     role='user', # Role might need adjustment depending on how it's processed
    #     parts=[
    #         types.Part.from_uri(
    #             file_uri=uri,
    #             # MIME type might be important for processing
    #             mime_type="application/pdf", # Assuming PDF, might need dynamic detection
    #         )
    #     ]
    # )
    # Returning a simple string confirmation instead of the Content object
    return f"File at URI {uri} acknowledged for session context."

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
    print(f"No existing RAG corpus found with display name '{corpa_name}'. Creating one now.")
    try:
        # Call the function to create the corpus and import files
        rag_corpus = create_rag_corpora(corpa_name, corpa_document_bucket)
        print(f"New RAG corpus creation initiated with name: {rag_corpus.name}")
        print("Note: File import and indexing may take some time to complete in the background.")
    except Exception as e:
        # Handle errors during corpus creation
        print(f"Error creating RAG corpus '{corpa_name}': {e}")
        rag_corpus = None # Ensure rag_corpus remains None on failure
else:
    # Corpus was found in the loop
    print(f"\nUsing existing RAG corpus: {rag_corpus.name}")

# Final check and confirmation message
if rag_corpus:
    print(f"\nProceeding with RAG corpus: {rag_corpus.name} (Display Name: {rag_corpus.display_name})")
    # Example test call to get_gcs_uri (can be removed if not needed for debugging)
    # test_uri_result = get_gcs_uri('How do I install a Nest E thermostat')
    # print(f"Test call to get_gcs_uri result: {test_uri_result}")
else:
    print(f"\nFailed to find or create RAG corpus '{corpa_name}'. RAG functionality will not work.")


# --- Sub-Agent Definitions ---
# @title Define RAG, Reasoning and Notes Sub-Agents

# --- RAG Agent ---
# This agent's role is to use the RAG tool (get_gcs_uri) to find relevant document URIs.
rag_agent = None
try:
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
    print(f"✅ Agent '{rag_agent.name}' created.")
except Exception as e:
    print(f"❌ Could not create RAG agent. Error: {e}")

# -- Reasoning Agent ---
# This agent's role is to generate troubleshooting steps based on document content
# (which is expected to be in the context after add_file_to_session is called).
reasoning_agent = None
try:
    reasoning_agent = Agent(
        #model="gemini-2.5-pro-exp-03-25", # Using a different model to account for potential resource constraints
        model="gemini-2.5-flash-preview-04-17",
        name="reasoning_agent",
        instruction=
        """
          You are a technical support specialist. Your task is to create a clear, step-by-step troubleshooting plan based on the provided support documents (which are now in your context).
          The user's original problem description will be provided.
          Analyze the information within the document(s) made available in this session context.
          If multiple documents were relevant, clearly state which document(s) contain the information used for the plan.
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
    print(f"✅ Agent '{reasoning_agent.name}' created.")
except Exception as e:
    print(f"❌ Could not create Reasoning agent. Error: {e}")

# --- Notes Agent ---
# This agent's role is to format the troubleshooting plan and add it as a note to the ticket system.
notes_agent = None
try:
    notes_agent = Agent(
        model="gemini-2.0-flash-001",
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
    print(f"✅ Agent '{notes_agent.name}' created.")
except Exception as e:
    print(f"❌ Could not create Notes agent. Error: {e}")

# --- Root Agent Definition ---
# @title Define the Root Agent with Sub-Agents

# Initialize root agent variables
root_agent = None
runner_root = None # Initialize runner variable (although runner is created later)

# Check if all necessary components (sub-agents and the specific tool) are available
if rag_agent and reasoning_agent and notes_agent and 'add_file_to_session' in globals():

    # Define the root agent (coordinator)
    nest_agent_team = Agent(
        name="nest_support_agent",    # Name for the root agent
        model="gemini-2.0-flash-001", # Model for the root agent (orchestration)
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
    print(f"✅ Root Agent '{root_agent.name}' created using model '{root_agent.model}' with sub-agents: {[sa.name for sa in root_agent.sub_agents]}")

else:
    # Print errors if sub-agents or tools were not initialized correctly
    print("❌ Cannot create root agent because one or more components are missing.")
    if not rag_agent: print(" - RAG Agent ('rag_agent') is missing.")
    if not reasoning_agent: print(" - Reasoning Agent ('reasoning_agent') is missing.")
    if not notes_agent: print(" - Notes Agent ('notes_agent') is missing.")
    if 'add_file_to_session' not in globals(): print(" - Tool 'add_file_to_session' function is missing.")
    root_agent = None # Ensure root_agent is None if creation failed

# --- Agent Interaction Execution ---
# @title Interact with the Agent Team

# Check if the root agent was successfully created in the previous step
if root_agent:
    # Define an async function to run the interactive conversation flow
    async def run_team_conversation():
        print("\n--- Starting Interactive Agent Session ---")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        # Use InMemorySessionService for simple, non-persistent conversation state management
        session_service = InMemorySessionService()

        # Define identifiers for the application, user, and session
        APP_NAME = "nest_support_agent_team_app" # An arbitrary name for the application context
        USER_ID = "interactive_user_01" # An identifier for the interactive user
        SESSION_ID = f"session_{os.urandom(8).hex()}" # Generate a unique session ID for each run

        # Create (or get) the session object using the service
        # This session will store the conversation history for the given user/session ID.
        session = session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        print(f"Session created: User='{USER_ID}', Session='{SESSION_ID}'")

        # --- Get the actual root agent object ---
        # (Already assigned to root_agent variable)

        # Create an ADK Runner instance for the root agent.
        # The runner manages the execution flow, tool calls, and sub-agent delegation.
        runner = Runner(
            agent=root_agent, # The root agent object orchestrates the interaction
            app_name=APP_NAME, # Associate the runner with the application context
            session_service=session_service # Provide the session service for state management
            # artifact_service can be added here if needed for file handling beyond basic context
        )
        print(f"Runner created for root agent '{root_agent.name}'. Ready for interaction.\n")

        # --- Interactive Conversation Loop ---
        while True:
            # Get user input from the console
            try:
                query = input("You: ")
            except EOFError: # Handle Ctrl+D or similar end-of-file signals
                print("\nExiting...")
                break

            # Check for exit commands
            if query.lower() in ["quit", "exit", "bye"]:
                print("Agent: Goodbye!")
                break

            # If input is empty, just loop again
            if not query.strip():
                continue

            # Call the agent with the user's query
            await call_agent_async(
                query=query,
                runner=runner,
                user_id=USER_ID,
                session_id=SESSION_ID
            )

    # --- Execute the asynchronous conversation ---
    # Use asyncio.run() to start the event loop and run the run_team_conversation function.
    # This initiates the interaction with the agent team.
    print("\nInitializing conversation...")
    try:
        asyncio.run(run_team_conversation())
    except KeyboardInterrupt: # Handle Ctrl+C gracefully
        print("\nConversation interrupted by user. Exiting.")
    print("\n--- Conversation Finished ---")

else:
    # Message if the root agent wasn't created successfully
    print("\n⚠️ Skipping agent team conversation as the root agent ('nest_agent_team' or 'root_agent') was not successfully defined.")

# --- End of Script ---