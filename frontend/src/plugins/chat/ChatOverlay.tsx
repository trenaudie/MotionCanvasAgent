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
  const [sessionId, setSessionId] = useState<string>('');
  const [userId, setUserId] = useState<string>(`user_${Date.now()}`);
  const [exampleCode, setExampleCode] = useState<string>('');
  const [consoleErrors, setConsoleErrors] = useState<string[]>([]);
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

  // Fetch example code from file
  const fetchExampleCode = async (): Promise<string> => {
    try {
      const response = await fetch('/src/scenes/example.tsx');
      if (response.ok) {
        const code = await response.text();
        setExampleCode(code);
        return code;
      } else {
        console.warn('Could not fetch example.tsx file');
        return '';
      }
    } catch (error) {
      console.warn('Error fetching example.tsx:', error);
      return '';
    }
  };

  // Fetch example code on component mount
  useEffect(() => {
    fetchExampleCode();
  }, []);

  // Capture console errors and runtime errors
  useEffect(() => {
    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    
    // Override console.error to capture errors
    console.error = (...args) => {
      const errorMessage = args.map(arg => 
        typeof arg === 'string' ? arg : JSON.stringify(arg)
      ).join(' ');
      
      // Filter for rendering/Motion Canvas related errors
      if (
        errorMessage.includes('kicker') ||
        errorMessage.includes('Scene2D') ||
        errorMessage.includes('runnerFactory') ||
        errorMessage.includes('Generator') ||
        errorMessage.includes('yield') ||
        errorMessage.includes('function') ||
        errorMessage.includes('TypeError') ||
        errorMessage.includes('ReferenceError') ||
        errorMessage.includes('motion-canvas') ||
        errorMessage.includes('.tsx') ||
        errorMessage.includes('example.tsx')
      ) {
        setConsoleErrors(prev => {
          const newErrors = [...prev, `ERROR: ${errorMessage}`].slice(-5); // Keep last 5 errors
          return newErrors;
        });
      }
      
      // Call original console.error
      originalConsoleError.apply(console, args);
    };

    // Override console.warn for warnings
    console.warn = (...args) => {
      const warnMessage = args.map(arg => 
        typeof arg === 'string' ? arg : JSON.stringify(arg)
      ).join(' ');
      
      if (
        warnMessage.includes('motion-canvas') ||
        warnMessage.includes('Scene2D') ||
        warnMessage.includes('.tsx')
      ) {
        setConsoleErrors(prev => {
          const newErrors = [...prev, `WARN: ${warnMessage}`].slice(-5);
          return newErrors;
        });
      }
      
      originalConsoleWarn.apply(console, args);
    };

    // Capture unhandled errors
    const handleError = (event: ErrorEvent) => {
      const errorMessage = `${event.message} at ${event.filename}:${event.lineno}:${event.colno}`;
      if (
        errorMessage.includes('kicker') ||
        errorMessage.includes('Scene2D') ||
        errorMessage.includes('motion-canvas') ||
        errorMessage.includes('.tsx')
      ) {
        setConsoleErrors(prev => {
          const newErrors = [...prev, `RUNTIME ERROR: ${errorMessage}`].slice(-5);
          return newErrors;
        });
      }
    };

    // Capture unhandled promise rejections
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      const errorMessage = `Unhandled Promise Rejection: ${event.reason}`;
      setConsoleErrors(prev => {
        const newErrors = [...prev, errorMessage].slice(-5);
        return newErrors;
      });
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      // Restore original console methods
      console.error = originalConsoleError;
      console.warn = originalConsoleWarn;
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  // Extract message sending logic into reusable function
  const sendMessageToBackend = async (messageText: string, includeContext: boolean = true) => {
    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      username,
      text: messageText,
      timestamp: new Date(),
      type: 'user'
    };
    setMessages(prev => [...prev, userMessage]);

    // Add status message
    const statusMessage: Message = {
      id: (Date.now() + 1).toString(),
      username: 'System',
      text: 'Fetching context and sending message to server...',
      timestamp: new Date(),
      type: 'system'
    };
    setMessages(prev => [...prev, statusMessage]);

    // Fetch the latest example code if context is needed
    const contextCode = includeContext ? await fetchExampleCode() : exampleCode;

    // Determine backend at runtime
    let backend = '';
    if (import.meta.env.BACKEND) {
      backend = import.meta.env.BACKEND;
    } else {
      backend = 'FLASK'; // Default to Flask if not set
    }

    // Prepare context message
    let contextualMessage = '';
    
    // Add current code context
    if (includeContext && contextCode) {
      contextualMessage += `Context - Current example.tsx content:\n\`\`\`typescript\n${contextCode}\n\`\`\`\n\n`;
    }
    
    // Add recent console errors if any and context is needed
    if (includeContext && consoleErrors.length > 0) {
      contextualMessage += `Recent Console Errors/Warnings:\n\`\`\`\n${consoleErrors.join('\n')}\n\`\`\`\n\n`;
    }
    
    // Add user request
    contextualMessage += `User request: ${messageText}`;
    
    // If no context, just use the message
    if (!includeContext || (!contextCode && consoleErrors.length === 0)) {
      contextualMessage = messageText;
    }

    if (backend === 'IBA') {
      // Stream fetch from IBA backend with correct request format
      try {
        const requestData = {
          message: contextualMessage,
          session_id: sessionId,
          user_id: userId,
          search_type: "hybrid"
        };

        const response = await fetch('http://0.0.0.0:8058/chat/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestData)
        });

        if (!response.body) throw new Error('No response body');

        // Remove status message
        setMessages(prev => prev.filter(msg => msg.id !== statusMessage.id));

        const reader = response.body.getReader();
        let assistantMessage = '';
        let toolsUsed: any[] = [];
        // Create assistant message placeholder
        const assistantMessageId = (Date.now() + 2).toString();
        const assistantMessageObj: Message = {
          id: assistantMessageId,
          username: 'Assistant',
          text: '',
          timestamp: new Date(),
          type: 'system'
        };
        setMessages(prev => [...prev, assistantMessageObj]);

        try {
          let done = false;
          while (!done) {
            const { value, done: streamDone } = await reader.read();
            done = streamDone;
            if (value) {
              const chunk = new TextDecoder().decode(value);
              const lines = chunk.split('\n');
              for (const line of lines) {
                const trimmedLine = line.trim();
                if (trimmedLine.startsWith('data: ')) {
                  try {
                    const data = JSON.parse(trimmedLine.slice(6));
                    if (data.type === 'session') {
                      setSessionId(data.session_id);
                    } else if (data.type === 'text') {
                      const content = data.content || '';
                      assistantMessage += content;
                      setMessages(prev => prev.map(msg => 
                        msg.id === assistantMessageId 
                          ? { ...msg, text: assistantMessage }
                          : msg
                      ));
                    } else if (data.type === 'tools') {
                      toolsUsed = data.tools || [];
                    } else if (data.type === 'end') {
                      done = true;
                    } else if (data.type === 'error') {
                      const errorContent = data.content || 'Unknown error';
                      throw new Error(errorContent);
                    }
                  } catch (jsonError) {
                    console.warn('Failed to parse JSON:', trimmedLine);
                  }
                }
              }
            }
          }
          if (toolsUsed.length > 0) {
            const toolsMessage: Message = {
              id: (Date.now() + 3).toString(),
              username: 'System',
              text: `Tools used: ${toolsUsed.map(tool => tool.name || tool).join(', ')}`,
              timestamp: new Date(),
              type: 'system'
            };
            setMessages(prev => [...prev, toolsMessage]);
          }
        } finally {
          reader.releaseLock();
        }
      } catch (error) {
        console.error('Error:', error);
        setMessages(prev => prev.filter(msg => msg.id !== statusMessage.id));
        const errorMessage: Message = {
          id: (Date.now() + 4).toString(),
          username: 'System',
          text: `Error sending message to IBA backend: ${error instanceof Error ? error.message : 'Unknown error'}. Is the server running on 0.0.0.0:8058?`,
          timestamp: new Date(),
          type: 'system'
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } else {
      // Flask backend
      try {
        // Send message with context to Flask server
        const response = await fetch('http://localhost:8000/write_code', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompt: contextualMessage })
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
    }
  };

  const sendMessage = async () => {
    if (!currentMessage.trim()) return;
    
    const messageText = currentMessage.trim();
    setCurrentMessage(''); // Clear input immediately
    
    await sendMessageToBackend(messageText);
  };

  const resolveErrors = async () => {
    if (consoleErrors.length === 0) return;

    // Compose an error resolution request message
    const errorResolutionMessage = `Please fix the following errors in my Motion Canvas code:

Errors encountered:
${consoleErrors.join('\n')}

Current code in example.tsx:
\`\`\`typescript
${exampleCode}
\`\`\`

Please provide the corrected code that resolves these errors.`;

    // Clear the console errors since we're addressing them
    setConsoleErrors([]);

    // Send the error resolution request
    await sendMessageToBackend(errorResolutionMessage, false); // Don't include context again since we're manually providing it
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

  const clearAllChats = async () => {
    // Clear chat messages
    setMessages([]);
    
    // Clear console errors
    setConsoleErrors([]);
    
    // Default Motion Canvas code
    const defaultCode = `import { makeScene2D } from '@motion-canvas/2d';

export default makeScene2D(function* (view) {
  view.fill('#000000'); 
});`;

    try {
      // Try to write directly to the file
      const response = await fetch('/src/scenes/example.tsx', {
        method: 'PUT',
        headers: {
          'Content-Type': 'text/plain',
        },
        body: defaultCode
      });

      if (response.ok) {
        // Update local state
        setExampleCode(defaultCode);
        
        // Add success message
        const successMessage: Message = {
          id: Date.now().toString(),
          username: 'System',
          text: 'Chat cleared and example.tsx automatically reset to default code.',
          timestamp: new Date(),
          type: 'system'
        };
        setMessages([successMessage]);
      } else {
        // If direct write fails, try alternative approach
        throw new Error('Direct file write not supported');
      }
    } catch (error) {
      console.warn('Direct file write failed, trying alternative approach:', error);
      
      // Alternative: Try using File System API if available
      try {
        if ('showSaveFilePicker' in window) {
          // This would require user interaction, so let's try a different approach
          throw new Error('File System API requires user interaction');
        } else {
          throw new Error('File System API not available');
        }
      } catch (fsError) {
        // Final fallback: Update local state and inform user
        setExampleCode(defaultCode);
        
        const warningMessage: Message = {
          id: Date.now().toString(),
          username: 'System',
          text: `Chat cleared and context reset. Please manually replace the content of example.tsx with:

\`\`\`typescript
${defaultCode}
\`\`\``,
          timestamp: new Date(),
          type: 'system'
        };
        setMessages([warningMessage]);
      }
    }
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
            Describe your motion object/scene <br /> 
            {exampleCode ? '✓ Context loaded' : '⚠ No context'} 
            {consoleErrors.length > 0 && (
              <span style={{ color: '#ffcc00' }}> • {consoleErrors.length} error(s)</span>
            )}
          </div>
          {consoleErrors.length > 0 && (
            <button
              onClick={resolveErrors}
              style={{
                fontSize: '10px',
                backgroundColor: '#ff6b6b',
                color: '#fff',
                border: 'none',
                borderRadius: '3px',
                padding: '2px 6px',
                marginTop: '4px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
              title="Automatically send errors to AI for resolution"
            >
              Resolve Error
            </button>
          )}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
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
          <button
            onClick={clearAllChats}
            style={{
              background: 'none',
              border: '1px solid #666',
              color: '#ff6b6b',
              fontSize: '10px',
              cursor: 'pointer',
              padding: '4px 6px',
              borderRadius: '4px',
              fontWeight: 'bold'
            }}
            title="Clear all chats and reset example.tsx"
          >
            CLEAR ALL
          </button>
        </div>
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