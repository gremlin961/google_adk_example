from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

# Data models for request bodies
class Ticket(BaseModel):
    ticket_id: int
    description: str
    customer_id: int
    contact_name: str

class Note(BaseModel):
    ticket_id: int
    contact_name: str
    note: str


# File paths (using os.path for better cross-platform compatibility)
TICKETS_FILE = os.path.join(os.path.dirname(__file__), "tickets.txt")
NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.txt")
print(f'Using file {NOTES_FILE} for notes.')


# Helper function to write data to a file
def write_to_file(filepath: str, data: str):
    try:
        with open(filepath, "a") as f:  # Open in append mode
            f.write(data + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to file: {e}")

@app.post("/ticket")
async def create_ticket(ticket: Ticket):
    """
    Creates a new ticket and appends it to the tickets.txt file.
    """
    ticket_data = (
        f"ticket_id: {ticket.ticket_id}, description: {ticket.description}, "
        f"customer_id: {ticket.customer_id}, contact_name: {ticket.contact_name}"
    )
    write_to_file(TICKETS_FILE, ticket_data)
    return {"message": f"Ticket {ticket.ticket_id} created successfully."}


@app.post("/notes")
async def add_note(note: Note):
    """
    Adds a note to the notes.txt file.
    """
    note_data = (
        f"ticket_id: {note.ticket_id}, contact_name: {note.contact_name}, note: {note.note}"
    )
    write_to_file(NOTES_FILE, note_data)
    return {"message": f"Note added for ticket {note.ticket_id}."}



# Example Usage (with uvicorn)
if __name__ == "__main__":
    import uvicorn

    # Create the files if they don't exist.
    if not os.path.exists(TICKETS_FILE):
        open(TICKETS_FILE, 'w').close()  # Create an empty file
    if not os.path.exists(NOTES_FILE):
         open(NOTES_FILE, 'w').close()

    uvicorn.run(app, host="0.0.0.0", port=8001)
