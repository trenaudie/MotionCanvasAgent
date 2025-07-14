import { makeScene2D } from '@motion-canvas/2d';
import { createRef, all, waitFor, tween } from '@motion-canvas/core';
import { Txt } from '@motion-canvas/2d';

export default makeScene2D(function* (view) {
  // Create a reference for the text
  view.fill('#000000'); // Set background color
  const helloText = createRef<Txt>();

  // Add the text to the view
  view.add(
    <Txt
      ref={helloText}
      text={'HELLO'}
      fontSize={60}
      fill={'white'}
      opacity={0} // Start invisible
    />
  );

  // Wait for a moment before starting the animation
  yield* waitFor(0.5);

  // Animate the text to appear and rotate
  yield* all(
    helloText().opacity(1, 1), // Fade in over 1 second
    tween(2, (value) => {
      helloText().rotation(value * Math.PI * 2); // Rotate full circle
    })
  );
});