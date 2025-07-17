import { Rect, makeScene2D } from '@motion-canvas/2d';
import { createRef, all, createSignal, waitFor, tween } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create a reference for the Eiffel Tower
  const eiffelTower = createRef<Rect>();

  // Create signals for the Eiffel Tower's position and scale
  const towerPositionX = createSignal(-view.width() / 2 + 100); // Position it on the left side
  const towerPositionY = createSignal(0);
  const towerScale = createSignal(1);

  // Add the Eiffel Tower (using a rectangle as a placeholder)
  view.add(
    <Rect
      ref={eiffelTower}
      x={towerPositionX}
      y={towerPositionY}
      width={50}
      height={200}
      fill={'#FFD700'} // Yellow color for the Eiffel Tower
    />
  );

  // Animate the Eiffel Tower's rotation and scaling
  yield* all(
    tween(2, (value) => {
      towerScale(value);
      eiffelTower().rotation(value * 360);
    }),
    towerPositionY(100, 2) // Move up while rotating
  );

  // Wait for a moment
  yield* waitFor(1);

  // Shrink the Eiffel Tower to zero
  yield* all(
    towerScale(0, 1), // Scale down to zero
    towerPositionY(0, 1) // Reset position to original
  );
});