import { Circle, makeScene2D } from '@motion-canvas/2d';
import { createRef, all, createSignal, waitFor } from '@motion-canvas/core';
import { Matrix } from 'mathjs';
import { Txt } from '@motion-canvas/2d/lib/components/Txt';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create references for the circle and the matrix
  const circleRef = createRef<Circle>();
  const matrixRef = createRef<Matrix>();

  // Create a signal for the circle's position
  const circleX = createSignal(0);
  const circleY = createSignal(0);

  // Add a circle to the view
  view.add(
    <Circle
      ref={circleRef}
      x={circleX}
      y={circleY}
      width={() => 100}
      height={() => 100}
      fill={'#FF0000'}
    />
  );

  // Add a static LaTeX matrix to the view
  view.add(
    <Txt
      ref={matrixRef}
      x={() => 200}
      y={() => 200}
      text={'\begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix}'}
      fontSize={30}
      fill={'#FFFFFF'}
    />
  );

  // Animate the circle's position
  yield* all(
    circleX(200, 2), // Move to the right
    circleY(100, 2)  // Move down
  );

  // Wait for a moment before finishing
  yield* waitFor(1);
});