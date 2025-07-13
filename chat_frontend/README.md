# Chat Frontend

A simple chatbox interface to interact with the Flask app.py server.

## Features

- Clean, modern chat interface
- One-directional communication (sends prompts to server, no response back)
- Auto-resizing textarea
- Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- Responsive design for mobile devices

## Usage

1. Make sure your Flask server (app.py) is running on localhost:5000
2. Open `index.html` in your web browser
3. Type your prompt in the textarea and click "Send" or press Enter
4. The message will be sent to the `/write_code` endpoint on your Flask server

## Files

- `index.html` - Main HTML structure
- `style.css` - Styling for the chat interface
- `script.js` - JavaScript for handling form submission and interactions

## Note

This is a simple one-way communication interface. Messages are sent to the server but no response is displayed back in the chat interface, as requested. 