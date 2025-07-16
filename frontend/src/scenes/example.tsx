import { Circle, Rect, Txt, makeScene2D } from '@motion-canvas/2d';
import { createRef, all, createSignal, waitFor } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');
  // Create refs for the school bus and text
  const schoolBus = createRef<Rect>();
  const busText = createRef<Txt>();

  // Create signals for the bus's position
  const busPositionX = createSignal(0);
  const busPositionY = createSignal(0);

  // Add the school bus (using a rectangle as a placeholder)
  view.add(
    <Rect
      ref={schoolBus}
      x={busPositionX}
      y={busPositionY}
      width={200}
      height={50}
      fill={'#FFD700'} // Yellow color for the school bus
    />
  );

  // Add text below the bus
  view.add(
    <Txt
      ref={busText}
      x={busPositionX}
      y={() => busPositionY() - 40} // Position text below the bus
      text={'🚌'} // School bus emoji
      fontSize={100}
      fill={'#FFFFFF'} // White color for the text
    />
  );

  // Animate the school bus's position
  yield* all(
    busPositionX(200, 2), // Move to the right
    busPositionY(100, 2)  // Move down
  );

  // Wait for a moment
  yield* waitFor(1);

  // Reset position
  yield* all(
    busPositionX(0, 2), // Move back to the left
    busPositionY(0, 2)  // Move back up
  );
});