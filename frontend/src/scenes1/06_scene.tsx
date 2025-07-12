import { Circle, makeScene2D } from '@motion-canvas/2d';
import { createRef } from '@motion-canvas/core';
import { easeOutBounce } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  const ball = createRef<Circle>();

  // Add the circle to the scene
  view.add(
    <Circle
      ref={ball}
      x={() => 0}
      y={() => 0}
      width={() => 50}
      height={() => 50}
      fill={'#4caf50'}
    />
  );

  // Animate the bouncing
  yield* ball().y(0, 0.5).ease(easeOutBounce);
  yield* ball().y(300, 1).ease(easeOutBounce);
  yield* ball().y(0, 1).ease(easeOutBounce);
});