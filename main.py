import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
import pdfplumber

# Load environment variables from a .env file
load_dotenv()

# Global variable to store extracted PDF content
pdf_content = "" 

# Load OpenAI API key and server port from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 
PORT = int(os.getenv('PORT', 5050))  # Default to port 5050 if not specified

def parse_pdf():
    """Parse all PDF files in the 'PDF' directory and extract their text content."""
    global pdf_content
    static_folder = os.path.join(os.getcwd(), "PDF")
    # Get list of all PDF files in the 'PDF' directory
    pdf_files = [f for f in os.listdir(static_folder) if f.endswith('.pdf')]

    # Extract text from each page of each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(static_folder, pdf_file)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pdf_content += page.extract_text()

# Parse PDF files on startup
parse_pdf()

# System message template for OpenAI API
SYSTEM_MESSAGE = f'You are a helpful AI assistant. If the question is not in this text, do not answer it:{pdf_content}'

# Voice settings for OpenAI API
VOICE = 'alloy'

# Event types to log during the WebSocket session
LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

# Create FastAPI application
app = FastAPI()

# Raise error if OpenAI API key is not set
if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=HTMLResponse)
async def index_page():
    """Return a basic message indicating the server is running."""
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio calls and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say("Please wait while we connect your call to the A. I. voice assistant.")
    
    # Extract host from the incoming request to form the WebSocket URL
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    
    # Return TwiML response as XML
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle bidirectional WebSocket communication between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    # Connect to OpenAI's WebSocket API for real-time audio processing
    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await send_session_update(openai_ws)
        stream_sid = None

        async def receive_from_twilio():
            """Receive audio data from Twilio and forward it to OpenAI."""
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        # Send audio data to OpenAI
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        # Log stream SID when a new stream starts
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive processed audio from OpenAI and send it back to Twilio."""
            nonlocal stream_sid
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)
                    if response['type'] == 'session.updated':
                        print("Session updated successfully:", response)
                    if response['type'] == 'response.audio.delta' and response.get('delta'):
                        # Send processed audio back to Twilio
                        try:
                            audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                            audio_delta = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_payload
                                }
                            }
                            await websocket.send_json(audio_delta)
                        except Exception as e:
                            print(f"Error processing audio data: {e}")
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        # Run both Twilio receive and send tasks concurrently
        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_session_update(openai_ws):
    """Send session update message to OpenAI to initialize audio stream settings."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},  # Enable server-side voice activity detection
            "input_audio_format": "g711_ulaw",  # Input audio format
            "output_audio_format": "g711_ulaw",  # Output audio format
            "voice": VOICE,  # Selected AI voice
            "instructions": SYSTEM_MESSAGE,  # Context instructions for AI
            "modalities": ["text", "audio"],  # Enable text and audio modalities
            "temperature": 0.8,  # Adjust AI response creativity
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

# Start the server using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
