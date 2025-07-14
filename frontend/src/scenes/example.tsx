import { makeScene2D } from '@motion-canvas/2d';
import { Circle, Node } from '@motion-canvas/2d';
import { createRef, all, waitFor, easeInOutExpo } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create a reference for the circle
  const movingCircle = createRef<Circle>();

  // Add the circle to the view
  view.add(
    <Circle
      ref={movingCircle}
      size={50}
      fill={'#ff0000'}
      x={() => 0}
      y={() => 0}
    />
  );

  // Animation loop to move the circle back and forth
  while (true) {
    // Move to the right
    yield* movingCircle().position.x(200, 1, easeInOutExpo);
    yield* waitFor(0.5);
    // Move to the left
    yield* movingCircle().position.x(-200, 1, easeInOutExpo);
    yield* waitFor(0.5);
  }
});