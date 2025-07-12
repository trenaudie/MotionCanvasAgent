import { makeScene2D } from '@motion-canvas/2d';
import { Circle } from '@motion-canvas/2d';
import { createRef } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  const circleRef = createRef<Circle>();

  // Add the circle at initial position
  view.add(<Circle ref={circleRef} x={() => -100} y={() => -200} radius={50} fill={'#4a90e2'} />);

  // Animate the circle bouncing up
  yield* circleRef().y(-200, 0.5).to(0, 1, { easing: easeOutElastic });
  // Animate bouncing back down
  yield* circleRef().y(0, 0.5).to(-200, 1, { easing: easeOutElastic });
});