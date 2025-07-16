import { Circle, makeScene2D } from '@motion-canvas/2d';
import { createRef } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create a reference for the red circle
  const redCircle = createRef<Circle>();

  // Add the red circle to the view
  view.add(
    <Circle
      ref={redCircle}
      x={() => 0}
      y={() => 0}
      width={() => 100}
      height={() => 100}
      fill={'#FF0000'} // Red color
    />
  );
});