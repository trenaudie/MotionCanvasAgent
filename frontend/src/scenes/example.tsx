import { Circle, makeScene2D } from '@motion-canvas/2d';
import { createRef, all, waitFor, easeInExpo } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create refs for the two circles
  const circle1 = createRef<Circle>();
  const circle2 = createRef<Circle>();

  // Add the circles to the view
  view.add(
    <>
      <Circle ref={circle1} x={() => -100} y={() => 0} width={100} height={100} fill={'red'} />
      <Circle ref={circle2} x={() => 100} y={() => 0} width={100} height={100} fill={'red'} />
    </>
  );

  // Wait for a moment before starting the animation
  yield* waitFor(0.5);

  // Animate the circles swapping positions
  yield* all(
    circle1().position.x(100, 1, easeInExpo),
    circle2().position.x(-100, 1, easeInExpo)
  );

  // Change the color of the circles to blue
  yield* all(
    circle1().fill('blue', 0.5),
    circle2().fill('blue', 0.5)
  );
});