import { Circle, makeScene2D } from '@motion-canvas/2d';
import { createRef } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create a reference for the circle
  const circleRef = createRef<Circle>();

  // Add a circle to the view
  view.add(
    <Circle
      ref={circleRef}
      x={() => 0} // Centered horizontally
      y={() => 0} // Centered vertically
      width={() => 100} // Diameter of the circle
      height={() => 100} // Diameter of the circle
      fill={'#FF0000'} // Fill color red
    />
  );
});