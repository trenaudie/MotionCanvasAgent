import { Rect, makeScene2D } from '@motion-canvas/2d';
import { createRef, all, createSignal, waitFor } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create a reference for the rectangle
  const rectRef = createRef<Rect>();

  // Create signals for the rectangle's position
  const rectX = createSignal(0);
  const rectY = createSignal(0);

  // Add a green rectangle to the view
  view.add(
    <Rect
      ref={rectRef}
      x={rectX}
      y={rectY}
      width={100}
      height={100}
      fill={'#00FF00'}
    />
  );

  // Animate the rectangle's position
  yield* all(
    rectX(200, 2), // Move to the right
    rectY(100, 2)   // Move down
  );

  // Wait for a moment before finishing
  yield* waitFor(1);
});