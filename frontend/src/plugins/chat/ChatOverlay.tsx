/* @jsxImportSource preact */
import { useState, useRef, useEffect } from 'preact/hooks';

interface Message {
  id: string;
  username: string;
  text: string;
  timestamp: Date;
  type: 'user' | 'system';
}

export function ChatOverlay() {
  const [isVisible, setIsVisible] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [username, setUsername] = useState('Anonymous');
  const [currentMessage, setCurrentMessage] = useState('');
  const [isEditingUsername, setIsEditingUsername] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const toggleChat = () => {
    setIsVisible(!isVisible);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!currentMessage.trim()) return;

    const messageText = currentMessage.trim();
    
    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      username,
      text: messageText,
      timestamp: new Date(),
      type: 'user'
    };
    setMessages(prev => [...prev, userMessage]);
    
    // Clear input
    setCurrentMessage('');

    // Add status message
    const statusMessage: Message = {
      id: (Date.now() + 1).toString(),
      username: 'System',
      text: 'Sending message to server...',
      timestamp: new Date(),
      type: 'system'
    };
    setMessages(prev => [...prev, statusMessage]);

    try {
      // Send message to Flask server
      const response = await fetch('http://localhost:8000/write_code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: messageText })
      });

      // Remove status message
      setMessages(prev => prev.filter(msg => msg.id !== statusMessage.id));

      if (response.ok) {
        const result = await response.text();
        const successMessage: Message = {
          id: (Date.now() + 2).toString(),
          username: 'System',
          text: 'Code generation completed successfully!',
          timestamp: new Date(),
          type: 'system'
        };
        setMessages(prev => [...prev, successMessage]);
      } else {
        const errorMessage: Message = {
          id: (Date.now() + 3).toString(),
          username: 'System',
          text: 'Failed to send message. Please try again.',
          timestamp: new Date(),
          type: 'system'
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Error:', error);
      
      // Remove status message
      setMessages(prev => prev.filter(msg => msg.id !== statusMessage.id));
      
      const errorMessage: Message = {
        id: (Date.now() + 4).toString(),
        username: 'System',
        text: 'Error sending message. Is the server running on localhost:8000?',
        timestamp: new Date(),
        type: 'system'
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    // Stop propagation to prevent Motion Canvas from receiving the event
    e.stopPropagation();
    
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
    // Allow other keys (including Tab) to work normally in the textarea
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const updateUsername = (newUsername: string) => {
    if (newUsername.trim() && newUsername !== username) {
      setUsername(newUsername.trim());
    }
    setIsEditingUsername(false);
  };

  if (!isVisible) {
    return (
      <div style={{
        position: 'fixed',
        bottom: '130px',
        left: '10px',
        zIndex: 9999
      }}>
        <button
          onClick={toggleChat}
          style={{
            width: '33px',
            height: '33px',
            borderRadius: '50%',
            backgroundColor: '#007acc',
            border: 'none',
            color: 'white',
            fontSize: '14px',
            cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          title="Open Chat"
        >
          💬
        </button>
      </div>
    );
  }

  return (
    <div 
      style={{
        position: 'fixed',
        bottom: '20px',
        left: '20px',
        width: '350px',
        height: '700px',
        backgroundColor: '#1e1e1e',
        color: '#ffffff',
        borderRadius: '12px',
        border: '1px solid #444',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
        zIndex: 9999
      }}
      onKeyDown={(e) => e.stopPropagation()}
      onKeyUp={(e) => e.stopPropagation()}
      onKeyPress={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div style={{
        padding: '16px',
        borderBottom: '1px solid #444',
        backgroundColor: '#2d2d2d',
        borderTopLeftRadius: '12px',
        borderTopRightRadius: '12px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '4px' }}>
            Motion Canvas Chat
          </div>
          <div style={{ fontSize: '12px', color: '#ffffff', backgroundColor: '#007acc', padding: '4px 8px', borderRadius: '4px' }}>
            Describe your motion object/scene <br /> and hit send
          </div>
        </div>
        <button
          onClick={toggleChat}
          style={{
            background: 'none',
            border: 'none',
            color: '#888',
            fontSize: '18px',
            cursor: 'pointer',
            padding: '4px'
          }}
          title="Close Chat"
        >
          ✕
        </button>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px'
      }}>
        {messages.map((message) => (
          <div
            key={message.id}
            style={{
              padding: '8px 12px',
              borderRadius: '8px',
              backgroundColor: message.type === 'system' ? '#2d4a2d' : '#333',
              border: message.type === 'system' ? '1px solid #4a7c4a' : '1px solid #555'
            }}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '4px'
            }}>
              <span style={{
                fontSize: '12px',
                fontWeight: 'bold',
                color: message.type === 'system' ? '#7fdf7f' : '#007acc'
              }}>
                {message.username}
              </span>
              <span style={{
                fontSize: '10px',
                color: '#888'
              }}>
                {formatTime(message.timestamp)}
              </span>
            </div>
            <div style={{
              fontSize: '13px',
              lineHeight: '1.4',
              whiteSpace: 'pre-wrap',
              color: '#ffffff'
            }}>
              {message.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: '12px',
        borderTop: '1px solid #444',
        backgroundColor: '#2d2d2d',
        borderBottomLeftRadius: '12px',
        borderBottomRightRadius: '12px'
      }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <textarea
            value={currentMessage}
            onChange={(e) => setCurrentMessage((e.target as HTMLTextAreaElement).value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            style={{
              flex: 1,
              minHeight: '36px',
              maxHeight: '80px',
              padding: '8px',
              border: '1px solid #666',
              borderRadius: '6px',
              backgroundColor: '#444',
              color: '#fff',
              fontSize: '13px',
              fontFamily: 'inherit',
              resize: 'vertical'
            }}
          />
          <button
            onClick={sendMessage}
            disabled={!currentMessage.trim()}
            style={{
              padding: '8px 16px',
              backgroundColor: currentMessage.trim() 
                ? '#007acc' 
                : '#555',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: currentMessage.trim() ? 'pointer' : 'not-allowed',
              fontSize: '13px',
              fontWeight: 'bold'
            }}
          >
            Send
          </button>
        </div>
        <div style={{
          fontSize: '10px',
          color: '#888',
          marginTop: '4px'
        }}>
          Enter to send • Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}
