import { makeScene2D } from '@motion-canvas/2d';
import { createRef, all, waitFor, createSignal } from '@motion-canvas/core';
import { Txt } from '@motion-canvas/2d';

export default makeScene2D(function* (view) {
  // Create a reference for the text
  const helloWorldText = createRef<Txt>();

  // Add the text to the view
  view.add(
    <Txt
      ref={helloWorldText}
      text={'Hello, World!'}
      fontSize={60}
      fill={'white'}
      opacity={0} // Start invisible
    />
  );

  // Wait for a moment before starting the animation
  yield* waitFor(0.5);

  // Animate the text to appear
  yield* all(
    helloWorldText().opacity(1, 1) // Fade in over 1 second
  );
});