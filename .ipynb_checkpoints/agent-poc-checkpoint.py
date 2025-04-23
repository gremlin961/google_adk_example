# Vertex AI Modules
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part, Tool, ChatSession, FunctionDeclaration, grounding, GenerationResponse
from vertexai.preview import rag

# Vertex Agent Modules
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.tools.agent_tool import AgentTool


# Vertex GenAI Modules
import google.genai
from google.genai import types

# Google Cloud AI Platform Modules
from google.cloud import aiplatform_v1beta1 as aiplatform # This module helps parse the info for delete_rag_corpa function
from google.cloud import storage

# Other Python Modules
#import base64
#from IPython.display import Markdown
import asyncio
import requests
import os
from typing import List, Dict, TypedDict, Any
import json
from urllib.parse import urlparse
import warnings
import logging


# Ignore all warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


project_id = "rkiles-demo-host-vpc" # Your GCP Project ID
location = "global" # You can leave this setting as global
region = "us-central1" # Your region. This notebook has only been tested in us-central1

corpa_name = "nest-rag-corpus" # This will be the display name of your RAG Engine corpus

corpa_document_bucket = "gs://rkiles-test/nest/docs/" # The GCS path to the files you want to ingest into your RAG Engine corpus

local_documents = "./nest_docs/" # Local directory containing Nest support files to copy

ticket_server_url = "http://ticket01:8000" # The url to the mock ticket system. This will be a GCE VM running the ticket_server.py web service.



os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = region



vertexai.init(project=project_id, location=region)



parsed_uri = urlparse(corpa_document_bucket)
bucket_name = parsed_uri.netloc
prefix = parsed_uri.path.lstrip('/')

storage_client = storage.Client()

# Get the bucket object
bucket = storage_client.bucket(bucket_name)

# Check if the bucket exists, create it if not
if not bucket.exists():
    bucket.create()
    print(f"Bucket '{bucket_name}' created successfully.")
else:
    print(f"Bucket '{bucket_name}' already exists.")

# Create the folder prefix if it doesn't implicitly exist
if prefix:
    blob_name = f"{prefix}" if prefix.endswith('/') else f"{prefix}/"
    placeholder_blob = bucket.blob(blob_name + ".placeholder")
    if not placeholder_blob.exists():
        placeholder_blob.upload_from_string('')
        print(f"Simulated folder '{corpa_document_bucket}' created.")
    else:
        print(f"Simulated folder '{corpa_document_bucket}' already exists.")

        
        
if os.path.exists(local_documents) and os.path.isdir(local_documents):
    for filename in os.listdir(local_documents):
        local_file_path = os.path.join(local_documents, filename)
        if os.path.isfile(local_file_path):
            gcs_blob_name = f"{prefix}{filename}"
            blob = bucket.blob(gcs_blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded '{local_file_path}' to 'gs://{bucket_name}/{gcs_blob_name}'")
else:
    print(f"Local directory '{local_documents}' does not exist or is not a directory.")
    
    
    
    
    
    

# @title Define Agent Interaction Function
import asyncio
from google.genai import types # For creating message Content/Parts

async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str): # <--- Added parameters
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response." # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    # Use the passed-in parameters now:
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")



    
    
    
    

def delete_rag_corpora(data: aiplatform.services.vertex_rag_data_service.pagers.ListRagCorporaPager):
    """Extracts 'name' values from a ListRagCorporaPager and prints them.

    Args:
        data: The ListRagCorporaPager object returned from the API.
    """

    names_list = []
    # First loop: Add names to the list
    for rag_corpus in data:  # Iterate through the rag_corpora objects
        if hasattr(rag_corpus, 'name'):  #  Check if the attribute exists
             names_list.append(rag_corpus.name)

    # Second loop: Print the names
    for corpa_name in names_list:
        rag.get_corpus(name=corpa_name)
        rag.delete_corpus(name=corpa_name)
        
        
        
        
def create_rag_corpora(display_name, source_bucket):
    EMBEDDING_MODEL = "publishers/google/models/text-embedding-004"  # @param {type:"string", isTemplate: true}
    embedding_model_config = rag.EmbeddingModelConfig(publisher_model=EMBEDDING_MODEL)

    rag_corpus = rag.create_corpus(
        display_name=display_name, embedding_model_config=embedding_model_config
    )
    

    
    INPUT_GCS_BUCKET = (
        source_bucket
    )

    response = rag.import_files(
        corpus_name=rag_corpus.name,
        paths=[INPUT_GCS_BUCKET],
        chunk_size=1024,  # Optional
        chunk_overlap=100,  # Optional
        max_embedding_requests_per_min=900,  # Optional
    )
    
    # This code shows how to upload local files to the corpus. 
    #rag_file = rag.upload_file(
    #    corpus_name=rag_corpus.name,
    #    path="./test.txt",
    #    display_name="test.txt",
    #    description="my test file"
    #)
    
    return rag_corpus


# @title Define Tools for creating a ticket, adding notes to a ticket, add a file to the session, and getting the GCS URI


def create_ticket(ticket_data: Dict) -> Dict:
    """
    Creates a ticket via the /ticket endpoint.

    Args:
        ticket_data: A dictionary containing the ticket data
                     (ticket_id, description, customer_id, contact_name).

    Returns:
        A dictionary containing the API response.  Handles errors gracefully.
    """
    url = f"{ticket_server_url}/ticket"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(ticket_data))
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    
    
    
    
    
def add_note(ticket_id: int, contact_name: str, note: str) -> Dict[str, Any]:
    """
    Adds a note to a ticket via the API's /notes endpoint.

    Sends the provided ticket details as JSON to the API and returns
    the parsed JSON response from the server.

    Args:
        ticket_id: int - The ID number of the ticket.
        contact_name: str - The name of the contact person.
        note: str - The content of the note to add.

    Returns:
         Dict[str, Any]:
            - On success: A dictionary representing the parsed JSON response
              from the API (content depends on the specific API implementation,
              often includes details of the created note or a success status).
            - On failure (request exception or non-2xx HTTP status):
              A dictionary containing an 'error' key with a description of the failure,
              e.g., {"error": "Request failed: 404 Client Error: Not Found for url: ..."}.
    """
    url = f"{ticket_server_url}/notes"
    headers = {"Content-Type": "application/json"}

    # Construct the dictionary *inside* the function
    note_data = {
        "ticket_id": ticket_id,
        "contact_name": contact_name,
        "note": note
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(note_data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}

    
    
    
    
    
#def add_file_to_session(uri: str) -> types.Content:
def add_file_to_session(uri: str) -> str:
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
    print(uri)
    content = types.Content(
        role='user',
        parts=[
            # Try passing ONLY the uri positionally, based on the error message "takes 1 positional argument"
            types.Part.from_uri(
                file_uri=uri,
                mime_type="application/pdf",
            )
        ]
    )

    return content





def get_gcs_uri(query: str) -> str:
    """
    Retrieves Google Cloud Storage (GCS) URIs for documents relevant to a given query.

    This function queries a pre-configured Retrieval-Augmented Generation (RAG)
    corpus to find documents related to the input query string. It extracts
    the source GCS URIs from the top relevant documents identified by the
    RAG system based on semantic similarity. Use this function when you need
    to find the source files in GCS that contain information related to a
    specific question or topic.

    Args:
        query: str - The natural language query or topic to search for within
                 the RAG corpus. For example: "What were the Q3 sales figures?"
                 or "Tell me about project Alpha's latest status".

    Returns:
         str - A JSON string representing a list of unique GCS URIs. These URIs
               point to the source documents found to be relevant to the query.
               Returns a JSON string representing an empty list ('[]') if no
               relevant documents meet the similarity criteria.
               Example return value: '["gs://my-bucket/doc1.pdf", "gs://my-bucket/report_q3.txt"]'
    """
    query_response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=rag_corpus.name,
                # Optional: supply IDs from `rag.list_files()`.
                # rag_file_ids=["rag-file-1", "rag-file-2", ...],
            )
        ],
        text=f'''
        {query}
        ''',
        similarity_top_k=10,  # Optional
        vector_distance_threshold=0.5,  # Optional
    )
    #print(response)
    uri_set = set()
    for context in query_response.contexts.contexts:
        uri_set.add(context.source_uri)
        #json.dumps(list(uri_set))
    #doc_uri = uri_set.pop()
    doc_uri = json.dumps(list(uri_set))
    return doc_uri





existing_corpora = rag.list_corpora()

print(existing_corpora)


# Variable to hold the corpus if found
found_corpus = None

# Iterate through all existing RAG corpora
for corpus in existing_corpora.rag_corpora: # Ensure you iterate the correct attribute
    # Check if display_name exists and matches
    if getattr(corpus, 'display_name', None) == corpa_name:
        print(f"Existing Corpa found. Using {corpus.name}")
        
        # You already have the corpus object, no need to call get_corpus usually
        # If 'corpus' object from the list is sufficient, use it directly.
        # If you MUST get a fresh object or different type, uncomment the next line:
        # rag_corpus = rag.get_corpus(name=corpus.name) 
        found_corpus = corpus # Store the found corpus object
        
        print(f"This corpus contains the following files:")
        try:
            # List files associated with the found corpus
            for file in rag.list_files(corpus.name): # Use corpus.name
                print(getattr(file, 'display_name', 'N/A')) # Safer access
        except Exception as e:
            print(f"Warning: Could not list files for {corpus.name}. Error: {e}")
            
        break # Exit the loop as soon as we find the match

# After the loop, check if we found anything
if found_corpus is None:
    # The loop completed without finding the corpus
    print(f"No existing {corpa_name} resource found. Creating one now.")
    try:
        rag_corpus = create_rag_corpora(corpa_name, corpa_document_bucket)
        print(f"New RAG corpus created at {rag_corpus.name}")
    except Exception as e:
        print(f"Error creating corpus {corpa_name}: {e}")
        rag_corpus = None # Indicate failure
else:
    # The corpus was found in the loop
    rag_corpus = found_corpus # Assign the found corpus to the main variable

# Now 'rag_corpus' holds either the found or newly created corpus (or None if creation failed)
# You can proceed to use 'rag_corpus' here
if rag_corpus:
    print(f"\nProceeding with corpus: {rag_corpus.name}")
    # ... your next steps using rag_corpus ...
else:
    print(f"\nFailed to find or create corpus '{corpa_name}'. Cannot proceed.")
    
    
    
    
    
test = get_gcs_uri('How do I install a Nest E thermostat')
print(test)

# @title Define RAG, Reasoning and Notes Sub-Agents

# --- RAG Agent ---
rag_agent = None
try:
    rag_agent = Agent(
        model="gemini-2.0-flash-001",
        name="rag_agent",
        instruction="""
          You are a customer support agent. You help locate documentation that will resolve customer issues.
          Identify the most relevant support document that pertains to the question.
          Your job is to only provide the GCS URI for the closest matching document, not to resolve the issue.
          You will use the get_gcs_uri to identify the correct file. 
          The response from get_gcs_uri will be a text string like 'gs://bucket_name/folder/file 1.pdf'
          Determine which files are relevant to the customer's question and return them to the root agent.
        """,
        description="Retrieves information from a RAG Engine instance and returns the GCS URI of relevant files.",
        tools=[
            get_gcs_uri
        ],
    )
    print(f"✅ Agent '{rag_agent.name}' created.")
except Exception as e:
    print(f"❌ Could not create Greeting agent. Error: {e}")


# -- Reasoning Agent ---
reasoning_agent = None
try:
    reasoning_agent = Agent(
        model="gemini-2.5-pro-exp-03-25",
        name="reasoning_agent",
        instruction="""
          You are a customer support agent. You help define the troubleshooting process to resolve customer issues.
          Use the information in the document to define the process for resolving the issue.
          Once the files have been added to your context, use that information to outline the process needed to resolve the identified problem.
          If multiple documents are provided, specify which document or documents contains the relevant information to resolve the issue.
          The process needs to outline the activities for the Nest technical support representitive to perform.
          The notes system only supports plain text. Ensure you only use text in your output.
      
          Example:
          Step 1: Do this
          Step 2: Do this other task
          Step 3: etc
        """,
        description="Defines the troubleshooting process to help resolve the customer's problem",
        tools=[
            #add_file_to_session,
        ],
    )
    print(f"✅ Agent '{reasoning_agent.name}' created.")
except Exception as e:
    print(f"❌ Could not create Greeting agent. Error: {e}")



# --- Notest Agent ---
notes_agent = None
try:
    notes_agent = Agent(
        model="gemini-2.0-flash-001",
        name="notes_agent",
        instruction="""
            You are a customer support assistant agent. Your primary task is to generate informative and well-structured notes for customer support tickets. 
            These notes will be added to the ticket's history and used by customer support representatives to understand the issue, track progress, and provide consistent service.
        """,
        description="Add notes to the associated ticket",
        tools=[
            add_note,
        ],
    )
    print(f"✅ Agent '{notes_agent.name}' created.")
except Exception as e:
    print(f"❌ Could not create Greeting agent. Error: {e}")



# @title Define the Root Agent with Sub-Agents

# Ensure sub-agents were created successfully before defining the root agent.
# Also ensure the original 'get_weather' tool is defined.
root_agent = None
runner_root = None # Initialize runner

if rag_agent and reasoning_agent and notes_agent and 'add_file_to_session' in globals():
    # Let's use a capable Gemini model for the root agent to handle orchestration
    root_agent_model = 'gemini-2.0-flash-001'
        
    nest_agent_team = Agent(
        name="nest_support_agent",
        model="gemini-2.0-flash-001",
        description="The main coordinator agent. Handles user requests and delegates tasks to specialists.",
        instruction="""
        You are a Nest customer support agent. You help triage and document actions for customer support tickets. 
        These notes will be added to the ticket's history and used by customer support representatives to understand the issue, track progress, and provide consistent service
        "You have specialized sub-agents: "
            "1. 'rag_agent': Handles retriving of relevant documents based on the user's question."
            "2. 'reasoning_agent': Handles the generation of troubleshooting steps based on the user's question and related documentation."
            "3. 'notes_agent': Adds information to the associated support ticket."
        Take the following actions to help the user resolve their problem and update the associated support ticket.
            Step 1: Start by identifying what the problem is, then call the rag_agent to identify the related documents.
            Step 2: Use the add_file_to_session tool to add the document to your context.The add_file_to_session tool only supports 1 document at a time. If the rag_agent provides multiple documents, you will need to make multiple calls using the add_file_to_session tool
            Step 3: Use the 'reasoning_agent' to help define the troubleshooting process. The reasoning_agent will return the process to troubleshoot the issue. 
            Step 4: Acknowledge you have the troubleshooting process and let the customer know you will update their ticket.
            Step 4: If the user has not provided you with their ticket ID and email, ask for it.
            Step 5: Use the 'notes_agent' to add the troubleshooting process to the ticket notes. The notes_agent is expecting the following json request body:
                {
                    ticket_id: int - A number representing the ticket ID number
                    contact_name: str - The name of the contact person as a string value
                    note: str - A string value that outlines the plan of action to resolve the ticket
                }
        """,
        tools=[
            add_file_to_session,
            AgentTool(agent=rag_agent), 
            AgentTool(agent=reasoning_agent)],
        sub_agents=[notes_agent]
    )
    print(f"✅ Root Agent '{nest_agent_team.name}' created using model '{root_agent_model}' with sub-agents: {[sa.name for sa in nest_agent_team.sub_agents]}")

else:
    print("❌ Cannot create root agent because one or more sub-agents failed to initialize or 'add_file_to_session' tool is missing.")
    if not rag_agent: print(" - RAG Agent is missing.")
    if not farewell_agent: print(" - Farewell Agent is missing.")
    if 'get_weather' not in globals(): print(" - get_weather function is missing.")




    
# @title Interact with the Agent Team

# Ensure the root agent (e.g., 'nest_agent_team' or 'root_agent' from the previous cell) is defined.
# Ensure the call_agent_async function is defined.

# Check if the root agent variable exists before defining the conversation function
root_agent_var_name = 'root_agent' # Default name from Step 3 guide
if 'nest_agent_team' in globals(): # Check if user used this name instead
    root_agent_var_name = 'nest_agent_team'
elif 'root_agent' not in globals():
    print("⚠️ Root agent ('root_agent' or 'nest_agent_team') not found. Cannot define run_team_conversation.")
    # Assign a dummy value to prevent NameError later if the code block runs anyway
    root_agent = None

if root_agent_var_name in globals() and globals()[root_agent_var_name]:
    async def run_team_conversation():
        print("\n--- Testing Agent Team Delegation ---")
        # InMemorySessionService is simple, non-persistent storage for this tutorial.
        session_service = InMemorySessionService()

        # Define constants for identifying the interaction context
        APP_NAME = "nets_support_agent_team"
        USER_ID = "user_1_agent_team"
        SESSION_ID = "session_001_agent_team" # Using a fixed ID for simplicity

        # Create the specific session where the conversation will happen
        session = session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

        # --- Get the actual root agent object ---
        # Use the determined variable name
        actual_root_agent = globals()[root_agent_var_name]

        # Create a runner specific to this agent team test
        runner = Runner(
            agent=actual_root_agent, # Use the root agent object
            app_name=APP_NAME,       # Use the specific app name
            session_service=session_service # Use the specific session service
            )
        # Corrected print statement to show the actual root agent's name
        print(f"Runner created for agent '{actual_root_agent.name}'.")

        # Always interact via the root agent's runner, passing the correct IDs
        await call_agent_async(query = "Hello there!", runner=runner, user_id=USER_ID, session_id=SESSION_ID)
        await call_agent_async(query = "I need to know how to setup my nest gen 3 unit.", runner=runner, user_id=USER_ID, session_id=SESSION_ID)
        await call_agent_async(query = "My name is John Doe and email is johnDoe@here.com. My ticket number is 132436.", runner=runner, user_id=USER_ID, session_id=SESSION_ID)
        await call_agent_async(query = "Great, thank you!", runner=runner, user_id=USER_ID, session_id=SESSION_ID)

    # Execute the conversation
    # Note: This may require API keys for the models used by root and sub-agents!
    asyncio.run(run_team_conversation())
else:
    print("\n⚠️ Skipping agent team conversation as the root agent was not successfully defined in the previous step.")
    
    
    
    


