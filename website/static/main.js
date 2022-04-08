//========================================================================
// Drag and drop image handling
//========================================================================

var fileDragFundus = document.getElementById("file-drag-fundus");
var fileDragOCT = document.getElementById("file-drag-oct");
var fileSelectFundus = document.getElementById("file-upload-fundus");
var fileSelectOCT = document.getElementById("file-upload-oct");

// Add event listeners
fileDragFundus.addEventListener("dragover", fileDragHoverFundus, false);
fileDragFundus.addEventListener("dragleave", fileDragHoverFundus, false);
fileDragFundus.addEventListener("drop", fileSelectHandlerFundus, false);
fileDragOCT.addEventListener("dragover", fileDragHoverOCT, false);
fileDragOCT.addEventListener("dragleave", fileDragHoverOCT, false);
fileDragOCT.addEventListener("drop", fileSelectHandlerOCT, false);

fileSelectFundus.addEventListener("change", fileSelectHandlerFundus, false);
fileSelectOCT.addEventListener("change", fileSelectHandlerOCT, false);

function fileDragHoverFundus(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDragFundus.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileDragHoverOCT(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDragOCT.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandlerFundus(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHoverFundus(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFileFundus(f);
  }
}

function fileSelectHandlerOCT(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHoverOCT(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFileOCT(f);
  }
}


//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreviewFundus = document.getElementById("image-preview-fundus");
var imagePreviewOCT = document.getElementById("image-preview-oct");
var imageDisplayFundus = document.getElementById("image-display-fundus");
var imageDisplayOCT = document.getElementById("image-display-oct");
var uploadCaptionFundus = document.getElementById("upload-caption-fundus");
var uploadCaptionOCT = document.getElementById("upload-caption-oct");
var predResultFundus = document.getElementById("pred-result-fundus");
var predResultOCT = document.getElementById("pred-result-oct");
var loaderFundus = document.getElementById("loader-fundus");
var loaderOCT = document.getElementById("loader-oct");
var predResultPrediction = document.getElementById("pred-result-prediction");

//========================================================================
// Main button events
//========================================================================

function submitImage() {
  // action for the submit button
  console.log("submit");

  if (!imageDisplayFundus.src || !imageDisplayFundus.src.startsWith("data")) {
    window.alert("Please select a fundus image before submitting.");
    return;
  }

  loaderFundus.classList.remove("hidden");
  loaderOCT.classList.remove("hidden");
  imageDisplayFundus.classList.add("loading");
  imageDisplayOCT.classList.add("loading");

  // call the predict function of the backend
  predictImage(imageDisplayFundus.src, imageDisplayOCT.src);
}

function reload() {
  location = self.location;
  return false;
}


function clearImage() {
  // reset selected files
  fileSelectFundus.value = "";
  fileSelectOCT.value = "";

  // remove image sources and hide them
  imagePreviewFundus.src = "";
  imagePreviewOCT.src = "";
  imageDisplayFundus.src = "";
  imageDisplayOCT.src = "";
  predResultFundus.innerHTML = "";
  predResultOCT.innerHTML = "";

  hide(imagePreviewFundus);
  hide(imagePreviewOCT);
  hide(imageDisplayFundus);
  hide(imageDisplayOCT);
  hide(loaderFundus);
  hide(loaderOCT);
  hide(predResultFundus);
  hide(predResultOCT);
  show(uploadCaptionFundus);
  show(uploadCaptionOCT);

  imageDisplayFundus.classList.remove("loading");
  imageDisplayOCT.classList.remove("loading");
}

function previewFileFundus(file) {
  // show the preview of the image
  console.log(file.name);
  var fileName = encodeURI(file.name);
  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreviewFundus.src = URL.createObjectURL(file);
    show(imagePreviewFundus);
    hide(uploadCaptionFundus);
    // reset
    predResultFundus.innerHTML = "";
    imageDisplayFundus.classList.remove("loading");
    displayImage(reader.result, "image-display-fundus");
  };
}

function previewFileOCT(file) {
  // show the preview of the image
  console.log(file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreviewOCT.src = URL.createObjectURL(file);
    show(imagePreviewOCT);
    hide(uploadCaptionOCT);
    // reset
    predResultOCT.innerHTML = "";
    imageDisplayOCT.classList.remove("loading");
    displayImage(reader.result, "image-display-oct");
  };
}

//========================================================================
// Helper functions
//========================================================================

function predictImage(image_fundus, image_oct) {
  // Call the predict function of the backend
  var indata = [image_fundus, image_oct];
  indata = JSON.stringify(indata);
  console.log("Data is in .JSON format: " + indata);
  
  // Send the image to the backend
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: indata
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          //data = Object.values(data);
          //var image_data = data[0];
          // Display the images
          //fundus_data = image_data[0];
          //oct_data = image_data[1];
          displayResult(data.result, element="image-display-prediction");
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data, element) { 
  console.log("running display result");
  console.log(data);

  displayImage(data, element);

  hide(imageDisplayFundus);
  hide(imageDisplayOCT);
  hide(loaderFundus);
  hide(loaderOCT);
  hide(predResultFundus);
  hide(predResultOCT);
  
}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}