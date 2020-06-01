/* See the blog post about BodyPix here: https://medium.com/tensorflow/introducing-bodypix-real-time-person-segmentation-in-the-browser-with-tensorflow-js-f1948126c2a0 */
/* This is a modified version of a bodypix example by Dan Oved https://twitter.com/oveddan */

let statusText = "";
// the downloaded bodyPix machine learning model
let model;
// the video capture
let capture;
// the most recent resulting mask image generated from estimating person segmentation on the video
let maskImage;
// the output canvas
let canvas;

let maskBackgroundButton;

let camShader;
let shaderLayer;
let copyLayer;

let w, h;

function preload() {
  // load the shader
  camShader = loadShader("uniform.vert", "uniform.frag");
}

function setup() {
  w = 1280;
  h = 1024;
  // save the created canvas so it can be drawn on with bodypix later.
  // createCanvas(windowWidth, windowHeight);
  canvas = createCanvas(w, h).canvas;

  loadModelAndStartEstimating();

  // capture from the webcam
  capture = createCapture(VIDEO);
  capture.hide();

  shaderLayer = createGraphics(w, h, WEBGL);
  shaderLayer.noStroke();

  copyLayer = createGraphics(w, h);
  copyLayer.noStroke();
}

/* the arguments to the function which draws the mask onto the canvas.  See the documentation for full descriptions:
https://github.com/tensorflow/tfjs-models/tree/master/body-pix#drawmask
*/
// opacity of the segmentation mask to be drawn on top of the image.
const maskOpacity = 1.0;
// if the output should be flip horizontally.  This should be set to true for user facing cameras.
const flipHorizontal = true;
// how much to blur the mask background by.  This affects the softness of the edge.
const maskBlurAmount = 3;

function draw() {
  background(255);

  textSize(16);
  text(statusText, 0, 30);

  // make sure video is loaded, and a mask has been estimated from the video.  The mask
  // continuously gets updated in the loop estimateFrame below, which is independent
  // from the draw loop
  if (capture && capture.loadedmetadata && maskImage) {
    const videoFrame = capture.get(0, 0, capture.width, capture.height);

    // use bodyPix to draw the estimated video with the most recent mask on top of it onto the canvas.
    bodyPix.drawMask(
      canvas,
      videoFrame.canvas,
      maskImage,
      maskOpacity,
      maskBlurAmount,
      flipHorizontal
    );
  }

  let c = get();

  shaderLayer.shader(camShader);

  camShader.setUniform("tex0", c);
  camShader.setUniform("tex1", copyLayer);
  camShader.setUniform("mouseDown", int(mouseIsPressed));
  camShader.setUniform("time", frameCount * 0.01);

  shaderLayer.rect(0, 0, w, h);

  copyLayer.image(shaderLayer, 0, 0, w, h);

  image(shaderLayer, 0, 0, w, h);
}

async function loadModelAndStartEstimating() {
  setStatusText("loading the model...");
  model = await bodyPix.load();

  setStatusText("");

  // start the estimation loop, separately from the drawing loop.
  // This allows drawing to happen at a high number of frame per
  // second, independent from the speed of estimation.
  startEstimationLoop();
}

function startEstimationLoop() {
  estimateFrame();
}

async function estimateFrame() {
  if (capture && capture.loadedmetadata) {
    await performEstimation();
  }

  // at the end of estimating, start again after the current frame is complete.
  requestAnimationFrame(estimateFrame);
}

/* the arguments to the functions which estimate the person segmentation and convert the results
to a mask to draw onto the canvas.  See the documentation for both methods:
https://github.com/tensorflow/tfjs-models/tree/master/body-pix#person-segmentation
https://github.com/tensorflow/tfjs-models/tree/master/body-pix#tomaskimagedata
*/
// set the output stride to 16 or 32 for faster performance but lower accuracy.
const outputStride = 8;
// affects the crop size around the person.  Higher number is tighter crop and visa
// versa for a lower number
const segmentationThreshold = 0.5;
// if the background or the person should be masked.  If set to false, masks the person.
const maskBackground = false;

async function performEstimation() {
  const videoFrame = capture.get(0, 0, capture.width, capture.height);
  const personSegmentation = await model.estimatePersonSegmentation(
    videoFrame.canvas,
    outputStride,
    segmentationThreshold
  );

  maskImage = bodyPix.toMaskImageData(
    personSegmentation,
    maskBackground,
    flipHorizontal
  );
}

function setStatusText(text) {
  statusText = text;
}
