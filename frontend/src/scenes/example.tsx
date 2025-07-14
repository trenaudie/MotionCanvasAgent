import { Circle, Rect, Triangle, makeScene2D, Txt } from '@motion-canvas/2d';
import { all, createRef, createSignal, waitFor } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create refs for the Circle, Triangle, and Rect
  const circle = createRef<Circle>();
  const triangle = createRef<Triangle>();
  const matrix = createRef<Rect>();

  // Add a Circle
  view.add(
    <Circle
      ref={circle}
      x={() => view.width() / 4}
      y={() => 0}
      width={() => 100}
      height={() => 100}
      fill={'#FF0000'}
    />
  );

  // Add a Triangle
  view.add(
    <Triangle
      ref={triangle}
      x={() => -view.width() / 4}
      y={() => 0}
      size={100}
      fill={'#00FF00'}
    />
  );

  // Add a static LaTeX matrix
  view.add(
    <Txt
      x={() => 0}
      y={() => -view.height() / 4}
      text={'\begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix}'}
      fontSize={40}
      fill={'#FFFFFF'}
    />
  );

  // Add a line
  view.add(
    <Rect
      x={() => -view.width() / 2}
      y={() => 0}
      width={() => view.width()}
      height={2}
      fill={'#FFFFFF'}
    />
  );

  // Wait for a moment before starting the animation
  yield* waitFor(1);

  // Animate the circle moving to the right
  yield* circle().position.x(view.width() / 2, 2);
  // Animate the triangle moving to the left
  yield* triangle().position.x(-view.width() / 2, 2);
});