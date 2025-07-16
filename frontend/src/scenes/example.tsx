import { Circle, Rect, Txt, makeScene2D } from '@motion-canvas/2d';
import { all, createRef, createSignal, waitFor } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create references for the circle, matrix, and line
  const circle = createRef<Circle>();
  const matrix = createRef<Rect>();
  const line = createRef<Rect>();

  // Create a signal for the circle's position
  const circleX = createSignal(0);
  const circleY = createSignal(0);

  // Add a circle to the view
  view.add(
    <Circle
      ref={circle}
      x={circleX}
      y={circleY}
      width={() => 50}
      height={() => 50}
      fill={'#FF5733'}
    />
  );

  // Add a static LaTeX matrix
  view.add(
    <Rect
      ref={matrix}
      x={() => -100}
      y={() => 100}
      width={() => 200}
      height={() => 100}
      fill={'#FFFFFF'}
    >
      <Txt
        text={'\begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix}'}
        fontSize={20}
        fill={'#000000'}
        x={() => -100}
        y={() => -50}
      />
    </Rect>
  );

  // Add a line
  view.add(
    <Rect
      ref={line}
      x={() => -150}
      y={() => -50}
      width={() => 300}
      height={() => 2}
      fill={'#FFFFFF'}
    />
  );

  // Animate the circle's position
  yield* all(
    circleX(-200, 2),
    circleY(100, 2)
  );

  yield* waitFor(1);

  // Move the circle back to the center
  yield* all(
    circleX(0, 2),
    circleY(0, 2)
  );
});