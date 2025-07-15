import {makeProject} from '@motion-canvas/core';
import example from './scenes/example?scene';

// Initialize chat integration
import { getChatInstance } from './plugins/chat/ChatIntegration';

// Auto-initialize chat when project loads
if (typeof window !== 'undefined') {
  console.log('Motion Canvas project loading, initializing chat...');
  // Delay initialization to ensure DOM is ready
  setTimeout(() => {
    console.log('Initializing chat instance...');
    try {
      const chatInstance = getChatInstance();
      console.log('Chat instance created:', chatInstance);
    } catch (error) {
      console.error('Error initializing chat:', error);
    }
  }, 1000);
}

export default makeProject({
  scenes: [example],
});
