import {Circle, makeScene2D, Path, Rect, Layout} from '@motion-canvas/2d';
import {createRef, all, waitFor} from '@motion-canvas/core';
import {logMethods} from './utils'; // Assuming this utility exists

export default makeScene2D(function* (view) {
  // Set the background color of the view
  view.fill('#000000');

  // Create refs for the Path, Circle, and the Rect that will serve as the bounding box
  const path = createRef<Path>();
  const circle = createRef<Circle>();
  const boundingBoxRect = createRef<Rect>(); // Ref for the bounding box Rect
  const boundingBoxCircle = createRef<Rect>();
  // Add the layout and its children to the view
  view.add(
    <Layout layout justifyContent={'center'} alignItems={'center'} gap={100}>
      {/* The Rect component for the bounding box */}
      <Rect
        ref={boundingBoxRect} // Assign the ref to this Rect
        stroke={'red'}        // Set stroke color to red
        lineWidth={2}         // Set line width
        fill={null}           // No fill
        // Initially set size and position to 0 or default values.
        // These will be updated once the Path's bounding box is calculated.
        size={[0, 0]}
      />
      <Path
        ref={path}            // Assign the ref to this Path
        lineWidth={4}
        stroke={'green'}
        data="M 151.34904,307.20455 L 264.34904,307.20455 C 264.34904,291.14096 263.2021,287.95455 236.59904,287.95455 C 240.84904,275.20455 258.12424,244.35808 267.72404,244.35808 C 276.21707,244.35808 286.34904,244.82592 286.34904,264.20455 C 286.34904,286.20455 323.37171,321.67547 332.34904,307.20455 C 345.72769,285.63897 309.34904,292.21514 309.34904,240.20455 C 309.34904,169.05135 350.87417,179.18071 350.87417,139.20455 C 350.87417,119.20455 345.34904,116.50374 345.34904,102.20455 C 345.34904,83.30695 361.99717,84.403577 358.75805,68.734879 C 356.52061,57.911656 354.76962,49.23199 353.46516,36.143889 C 352.53959,26.857305 352.24452,16.959398 342.59855,17.357382 C 331.26505,17.824992 326.96549,37.77419 309.34904,39.204549 C 291.76851,40.631991 276.77834,24.238028 269.97404,26.579549 C 263.22709,28.901334 265.34904,47.204549 269.34904,60.204549 C 275.63588,80.636771 289.34904,107.20455 264.34904,111.20455 C 239.34904,115.20455 196.34904,119.20455 165.34904,160.20455 C 134.34904,201.20455 135.49342,249.3212 123.34904,264.20455 C 82.590696,314.15529 40.823919,293.64625 40.823919,335.20455 C 40.823919,353.81019 72.349045,367.20455 77.349045,361.20455 C 82.349045,355.20455 34.863764,337.32587 87.995492,316.20455 C 133.38711,298.16014 137.43914,294.47663 151.34904,307.20455 z"
        scale={0.5}
        start={0}
        end={0}
      ></Path>
    </Layout>
  );

  // Wait for 1 second to ensure all elements are rendered and their properties calculated.
  yield* waitFor(1);

  // Now that the path is rendered, we can safely get its bounding box.
  const bbox = path().getCacheBBox();
  // const bboxCircle = circle().getCacheBBox();
  // Update the position and size of the boundingBoxRect using the calculated bbox.
  // The '0' duration means the change happens instantly.
  yield* boundingBoxRect().position(bbox.center, 0);
  console.log('bbox.position', bbox.center.x, bbox.center.y);
  console.log('bbox absolute position', bbox.center.transformAsPoint(boundingBoxRect().localToParent()));
  console.log('path absolute position', path().position().transformAsPoint(path().localToParent()));
  console.log('path position', path().position().x, path().position().y);
  yield* boundingBoxRect().size(bbox.size, 0);
  // yield* boundingBoxCircle().position(bboxCircle.center, 0);
  // console.log('bboxCircle.position', bboxCircle.center.x, bboxCircle.center.y);
  // console.log('circle position', circle().position().x, circle().position().y);
  // yield* boundingBoxCircle().size(bboxCircle.size, 0);


  // Animate the drawing of the circle and path
  // yield* all(...[circle().end(1, 1), path().end(1, 1)]);
  yield* path().end(1, 1);


  // Animate filling the circle and path with green color
  // yield* circle().fill('green', 2);
  yield* path().fill('green', 1);
});
