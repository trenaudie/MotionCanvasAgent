
import { makeScene2D, Circle, Rect, Txt, Line } from '@motion-canvas/2d';
import { createRef, all, createComputed } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#FFFFFF');

  // Create refs for the Circle, Rect, and Line
  const circle = createRef<Circle>();
  const matrixRect = createRef<Rect>();
  const line = createRef<Line>();

  // Create dynamic values for the circle's position
  const circleX = createComputed(() => view.width() / 2);
  const circleY = createComputed(() => view.height() / 2);

  // Add the Circle to the view
  view.add(
    <Circle
      ref={circle}
      x={circleX}
      y={circleY}
      radius={50}
      fill={'#FF0000'}
    />
  );

  // Add a static LaTeX matrix as a Rect
  view.add(
    <Rect
      ref={matrixRect}
      x={circleX}
      y={circleY + 100}
      width={200}
      height={100}
      fill={'#FFFFFF'}
      stroke={'#000000'}
      lineWidth={2}
    >
      <Txt
        text={`\\begin{bmatrix} 1 & 2 \\\ 3 & 4 \\end{bmatrix}`}
        fontSize={20}
        fill={'#000000'}
        x={0}
        y={0}
      />
    </Rect>
  );

  // Add a Line below the matrix
  view.add(
    <Line
      ref={line}
      x={circleX - 100}
      y={circleY + 150}
      points={[{ x: 0, y: 0 }, { x: 200, y: 0 }]}
      stroke={'#000000'}
      lineWidth={2}
    />
  );

  // Animate the circle's position
  yield* all(
    circle().position(circleX(), circleY() - 50, 1),
    circle().fill('#00FF00', 1)
  );
});