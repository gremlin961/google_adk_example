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

#Google BigQuery
from google.cloud import bigquery
from google.cloud.bigquery.schema import SchemaField 
from google.api_core.exceptions import GoogleAPICallError, BadRequest, NotFound, Forbidden



# Other Python Modules
#import base64
#from IPython.display import Markdown
import asyncio
import requests
import os
from typing import List, Dict, TypedDict, Any, Optional
import json
from urllib.parse import urlparse
import warnings
import logging
import sys
import traceback # For detailed error logging


# Ignore all warnings
warnings.filterwarnings("ignore")
# Set logging level to ERROR to suppress informational messages
logging.basicConfig(level=logging.ERROR)

# --- Configuration ---
project_id = "YOUR_PROJECT_ID" # Your GCP Project ID
location = "global" # You can leave this setting as global
region = "us-central1" # Your region. This notebook has only been tested in us-central1

bq_project = project_id
bq_dataset = "Product_Inventory"
bq_table = "product_data"



# --- Environment Setup ---
# Set environment variables required by some Google Cloud libraries
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1" # Instructs the google.genai library to use Vertex AI backend
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = region

# --- Initialize Vertex AI SDK ---
# Initialize the Vertex AI client library with project and location/region details
vertexai.init(project=project_id, location=region)







# --- Agent Tool Definitions ---
# @title Define Tools for defining the schema and executing a query

# @title Define function for collecting the schema of a table
async def get_bq_schema(
    bq_project_id: str,
    bq_dataset_id: str,
    bq_table_id: str
) -> Dict[str, Any]:
    """
    Gets the schema of a BigQuery table directly using the BigQuery client library,
    handling nested and repeated fields.

    Args:
        bq_project_id: The Google Cloud project ID containing the BigQuery table.
        bq_dataset_id: The BigQuery dataset ID.
        bq_table_id: The BigQuery table ID.

    Returns:
        A dictionary containing either:
          {'schema': [{'name': str, 'type': str, 'mode': str,
                       'description': Optional[str], 'fields': Optional[List[Dict]]}, ...]} on success.
                       The 'fields' key is present only for RECORD types.
          {'error': str} on failure
    """
    full_table_name = f"{bq_project_id}.{bq_dataset_id}.{bq_table_id}"
    try:
        # --- Helper function to process fields recursively ---
        def _process_field(field: SchemaField) -> Dict[str, Any]:
            """Converts a SchemaField to a dictionary, handling nesting."""
            field_dict = {
                "name": field.name,
                "type": field.field_type, # Use field_type consistently
                "mode": field.mode,
                "description": field.description # Can be None
            }
            # If the field is a RECORD (STRUCT), process its sub-fields recursively
            if field.field_type == "RECORD" or field.field_type == "STRUCT": # Handle both common names
                # Ensure field.fields is not None before iterating
                nested_fields = []
                if field.fields:
                    nested_fields = [_process_field(sub_field) for sub_field in field.fields]
                field_dict["fields"] = nested_fields

            # Clean up None description if desired (optional)
            # if field_dict["description"] is None:
            #    del field_dict["description"]
            return field_dict
        # --- End of helper function ---

        print(f"Attempting to get schema for: {full_table_name}")
        client = bigquery.Client(project=bq_project_id)
        table_ref = client.dataset(bq_dataset_id).table(bq_table_id)
        table = client.get_table(table_ref) # API request

        # Use the helper function to process the schema
        # The SchemaField object uses 'field_type' attribute, not 'type'
        # The 'fields' attribute holds sub-fields for RECORD types
        schema_list = [_process_field(field) for field in table.schema]

        print(f"Successfully retrieved schema for {full_table_name}")
        return {"schema": schema_list} # Return dictionary with schema list

    except NotFound:
        error_msg = f"BigQuery table {full_table_name} not found."
        print(f"ERROR: get_bq_schema: {error_msg}")
        return {"error": f"Error: {error_msg}"}
    except Exception as e:
        print(f"ERROR: get_bq_schema: An unexpected error occurred for {full_table_name}: {e}")
        traceback.print_exc()
        return {"error": f"Error: Failed to get schema from BigQuery API for {full_table_name}. Error type: {type(e).__name__}"}


# @title Define function for executing a SQL query
async def execute_bq_query(
    bq_project_id: str,
    bq_query: str, # <<< Renamed parameter
    max_results: Optional[int] = 1000
) -> Dict[str, Any]:
    """
    Executes a BigQuery SQL query (potentially generated by an LLM)
    and returns the results or status.

    Args:
        bq_project_id: The Google Cloud project ID where the query should run.
        bq_query: The SQL query string to execute.  (<<< Updated docstring)
        max_results: The maximum number of rows to return. Defaults to 1000.
                     Set to None to remove the limit (use with caution!).

    Returns:
        A dictionary containing:
        - {'results': list_of_dictionaries} where each dictionary represents a row,
          if the query is a SELECT statement and returns rows.
        - {'message': 'Query executed successfully, no rows returned.'} if the query
          is a SELECT statement that returns no rows.
        - {'message': 'Query executed successfully (non-SELECT statement).'} if the
          query is not a SELECT statement (e.g., INSERT, UPDATE, CREATE).
            - {'error': 'Error message description'} if an error occurs during execution.

    Raises:
        Nothing directly, but logs errors and returns them in the dictionary.

    Security Note: This function executes arbitrary SQL. Ensure the executing
                   service account has least-privilege permissions.
    Cost Note: Queries may incur BigQuery costs based on data processed.
    """
    print(f"Attempting to execute query in project {bq_project_id}:")
    # <<< Updated variable name in log message
    print(f"Query (first 500 chars): {bq_query[:500]}{'...' if len(bq_query) > 500 else ''}")
    print(f"Max results: {max_results}")

    try:
        client = bigquery.Client(project=bq_project_id)

        job_config = bigquery.QueryJobConfig()
        # Start the query job.
        # <<< Updated variable name in client.query call
        query_job = client.query(bq_query, job_config=job_config)

        # Wait for the job to complete.
        print(f"Waiting for query job {query_job.job_id} to complete...")
        results_iterator = query_job.result()
        print(f"Query job {query_job.job_id} completed. Status: {query_job.state}")

        # Check if it was a SELECT-like statement
        if query_job.statement_type == "SELECT":
             # Process results for SELECT queries
             rows_list = []
             count = 0
             for row in results_iterator:
                  rows_list.append(dict(row.items()))
                  count += 1
                  if max_results is not None and count >= max_results:
                       print(f"Reached max_results limit ({max_results}). Truncating results.")
                       break

             print(f"Returning {len(rows_list)} rows.")
             if not rows_list and results_iterator.total_rows == 0:
                  return {"message": "Query executed successfully, no rows returned."}
             else:
                 response = {"results": rows_list}
                 if max_results is not None and results_iterator.total_rows is not None and results_iterator.total_rows > max_results:
                     response["metadata"] = {"warning": f"Result truncated. Only the first {max_results} rows out of {results_iterator.total_rows} are included."}
                 elif results_iterator.total_rows is not None:
                     response["metadata"] = {"total_rows_processed_or_returned": results_iterator.total_rows}

                 return response

        else:
            # For non-SELECT statements
            statement_type = query_job.statement_type or "Unknown DML/DDL"
            print(f"Query was a non-SELECT statement ({statement_type}).")
            affected_rows_msg = ""
            if query_job.num_dml_affected_rows is not None:
                affected_rows_msg = f" Affected rows: {query_job.num_dml_affected_rows}."

            return {"message": f"Query executed successfully ({statement_type}).{affected_rows_msg}"}

    # <<< Error handling remains the same >>>
    except BadRequest as e:
        error_msg = f"Invalid Query or Bad Request: {e}"
        print(f"ERROR: execute_bq_query: {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}
    except NotFound as e:
        error_msg = f"Not Found Error (e.g., table does not exist): {e}"
        print(f"ERROR: execute_bq_query: {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}
    except Forbidden as e:
        error_msg = f"Permission Denied: {e}"
        print(f"ERROR: execute_bq_query: {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}
    except GoogleAPICallError as e:
        error_msg = f"BigQuery API Call Error: {e}"
        print(f"ERROR: execute_bq_query: {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred: {type(e).__name__} - {e}"
        print(f"ERROR: execute_bq_query: {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}

print("Agent interaction function `execute_bq_query` defined (using 'bq_query' parameter).")




# --- Sub-Agent Definitions ---
# @title Define RAG, Reasoning and Notes Sub-Agents

# --- SQL Agent ---
sql_agent = None

sql_agent = Agent(
    model="gemini-2.0-flash-001",
    name="sql_agent",
    instruction=
    f"""
        You are a SQL expert that inspects a given BigQuery table to find its schema. 
                
        You have access to specialized tools and sub-agents:
        
        1. Tool `get_bq_schema`: Use this tool to better understand the BigQuery tables schema. This information will be used to help the user write SQL statements to resolve their questions.
        Use the following informaiton when using the `get_bq_schema` tool:
            bq_project_id = {bq_project}
            bq_dataset_id = {bq_dataset}
            bq_table_id = {bq_table}
        2. Tool `execute_bq_query`: Use this tool to execute the SQL query against the BigQuery table. 
        
        Remember to use LOWER() functions when using operators such as 'LIKE' to ensure search reliability.

        An example workflow would be:
        Step 1: Use the `get_bq_schema` tool to retrieve the table schema information.
        Step 1: Using the schema information, write the SQL query.
        Step 3: Use the `execute_bq_schema` tool to execute the SQL query and summarize the result to the user.
    """,
    description="Retrieves schema information from a BigQuery table.",
    tools=[
        get_bq_schema,
        execute_bq_query
    ],
)


# --- Marketing Agent ---
marketing_agent = None
marketing_agent = Agent(
    model="gemini-2.0-flash-001",
    name="marketing_agent",
    instruction=
    """
        You are a marketing professional that creates product overviews. 
                
        Generate a marketing brochure for the product based on the supplied inventory data

        You can use the `sql_agent` AgentTool to retireve additional product data if you don't already have it.
    """,
    description="Generates marketing content.",
    tools=[
        AgentTool(agent=sql_agent),
    ],
)



# --- Root Agent Definition ---
# @title Define the Root Agent with Sub-Agents

# @title Define the Root Agent with Sub-Agents

# Ensure sub-agents were created successfully before defining the root agent.
# Also ensure the original 'get_weather' tool is defined.
root_agent = None
runner_root = None # Initialize runner

if sql_agent and marketing_agent and 'get_bq_schema' and 'execute_bq_query' in globals():
    # Let's use a capable Gemini model for the root agent to handle orchestration
    root_agent_model = 'gemini-2.0-flash-001'
        
    support_agent_team = Agent(
        name="support_agent",
        model="gemini-2.0-flash-001",
        description="The main coordinator agent. Handles user requests and delegates tasks to specialists.",
        instruction=
        f"""
            You are the lead customer support coordinator agent. Your goal is to understand the customer's issue, find relevant information from a BigQuery database.

            You have access to specialized tools and sub-agents:
            
            Use the following informaiton when `sql_agent` AgentTool:
                bq_project_id = {bq_project}
                bq_dataset_id = {bq_dataset}
                bq_table_id = {bq_table}
            
            Sub-Agent: 
            1. `sql_agent' - Collects SQL schemas and performs SQL queries against BigQuery tables.
            2. 'marketing_agent' - Generates professional marketing material based on the product details retrieved from the sql_agent.
            

            An example workflow:
            Step 1: Ask the user what they want to know from the database
            Step 2: Use the `sql_agent` AgentTool to retireve the information from the product database
            Step 3: Handoff to the `marketing_agent` to generate marketing content using the data from the sql_agent.
            """,
        tools=[
            #get_bq_schema,
            #execute_bq_query,
            AgentTool(agent=sql_agent), 
            #AgentTool(agent=marketing_agent)
        ],
        sub_agents=[marketing_agent]
    )

    root_agent = support_agent_team

# --- Agent Interaction Execution ---
# @title Interact with the Agent Team

