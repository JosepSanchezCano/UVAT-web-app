const modosAnotador = {
  MODELO: "MODELO",
  CORRECION: "CORRECION"
}

// Variables del anotador
var modo = modosAnotador.MODELO
var puntos = Array()
var all_masks = Array()
// Canvas and video variables
var canvas = document.getElementById("myCanvas"); // get the canvas from the page
var ctx = canvas.getContext("2d");
var videoContainer; // object to hold video and associated info
var video = document.getElementById("video"); // create a video element
var currentFrame = 0
var counter = 0
var playNextFrame = false
var fps = 15.049

var videoFrames = Array()

var drawnVideoSize = (0,0)
var playingForward = true

video.autoPlay = false; // ensure that the video does not auto play
video.loop = false; // set the video to loop.
videoContainer = {  // we will add properties as needed
     video : video,
     ready : false,   
};
var verticalPadding = 0
ctx.fillStyle = "black"

// To handle errors. This is not part of the example at the moment. Just fixing for Edge that did not like the ogv format video
video.onerror = function(e){
    //  document.body.removeChild(canvas);
    document.body.innerHTML += "<h2>There is a problem loading the video</h2><br>";
    document.body.innerHTML += "Users of IE9+ , the browser does not support WebM videos used by this demo";
    document.body.innerHTML += "<br><a href='https://tools.google.com/dlpage/webmmf/'> Download IE9+ WebM support</a> from tools.google.com<br> this includes Edge and Windows 10";
    
 }
video.oncanplay = readyToPlayVideo; // set the event to the play function that 
                                  // can be found below
function readyToPlayVideo(event){ // this is a referance to the video
    // the video may not match the canvas size so find a scale to fit
    videoContainer.scale = Math.min(
                         canvas.width / this.videoWidth, 
                         canvas.height / this.videoHeight); 
    videoContainer.ready = true;
     
    // the video can be played so hand it off to the display function
    requestAnimationFrame(updateCanvas);
    createPointData();
    // add instruction
    // document.getElementById("playPause").textContent = "Click video to play/pause.";
    // document.querySelector(".mute").textContent = "Mute";
}

function calcularFrameActual(video) {
  // currentTime devuelve el tiempo actual del video en segundos
  return Math.floor(video.currentTime * fps);
}

function stepForward() {
  video.pause();
  video.currentTime += 1 / fps; // Adjust for video frame rate
  video.addEventListener("seeked", updateCanvas, { once: true });
}


function stepBackward() {
  video.pause();
  video.currentTime -= 1 / fps; // Adjust for video frame rate
  video.addEventListener("seeked", updateCanvas, { once: true });
}


function loadVideo(){
  getFrames();
  console.log("loading video")
  console.log(videoFrames)
  ctx.drawImage(videoFrames[0], 0, 0, 1200, 800);
  }
function updateCanvas(){
    ctx.clearRect(0,0,canvas.width,canvas.height); 
    // only draw if loaded and ready
    if(videoContainer !== undefined && videoContainer.ready){ 
        // // find the top left of the video on the canvas
        // // video.muted = muted;
        // var scale = videoContainer.scale;
        // var vidH = videoContainer.video.videoHeight;
        // var vidW = videoContainer.video.videoWidth;
        // // var top = canvas.height / 2 - (vidH /2 ) * scale;
        // // var left = canvas.width / 2 - (vidW /2 ) * scale;
        // var top = 0;
        // var left= 0;
        // now just draw the video the correct size

        


        ctx.drawImage(videoContainer.video, left, top, vidW * scale, vidH * scale);
        // currentFrame = calcularFrameActual(videoContainer.video)
        // console.log(currentFrame)
        drawnVideoSize = [Math.floor(vidW * scale), Math.floor(vidH * scale)]
        verticalPadding = Math.floor((canvas.height - (vidH * scale)) / 2)
        drawMasks()
        if(puntos[currentFrame].length > 0){
          console.log("checking points")
          drawPoints(ctx)
          
        }

        if(playNextFrame){
          if(counter == 0){  
            counter += 1
          }else{
            videoContainer.video.pause()
            playNextFrame=false
            counter = 0  
          }
        }
        // if(videoContainer.video.paused){ // if not playing show the paused screen 
        //     drawPayIcon();
        // }

    }
    // all done for display 
    // request the next frame in 1/60th of a second
    // // console.log("Requesting frame")
    setTimeout(() => {
      if((!videoContainer.video.paused && !videoContainer.video.ended )){
        if (playingForward){
          requestAnimationFrame(updateCanvas);
  
        }else{
          video.currentTime -= 1 / fps; 
          requestAnimationFrame(updateCanvas);
        }

      }
    }, 1000 / fps);



    // requestAnimationFrame(updateCanvas);
}

function clearAllMasks(){

  $.ajax({
    url:"/clear_masks",
    type:"POST",
    data: {},
    success: function(response){
      all_masks = []
    },
    error: function(error){
      console.log(error);
    },
  
    });
}

function saveAnnotations(){
  $.ajax({
    url:"/save_ann",
    type:"POST",
    data: {},
    success: function(response){
      alert("Saved annotations locally.")
    },
    error: function(error){
      console.log(error);
    },
  
    });
}

function drawPayIcon(){
     ctx.fillStyle = "black";  // darken display
     ctx.globalAlpha = 0.5;
     ctx.fillRect(0,0,canvas.width,canvas.height);
     ctx.fillStyle = "#DDD"; // colour of play icon
     ctx.globalAlpha = 0.75; // partly transparent
     ctx.beginPath(); // create the path for the icon
     var size = (canvas.height / 2) * 0.5;  // the size of the icon
     ctx.moveTo(canvas.width/2 + size/2, canvas.height / 2); // start at the pointy end
     ctx.lineTo(canvas.width/2 - size/2, canvas.height / 2 + size);
     ctx.lineTo(canvas.width/2 - size/2, canvas.height / 2 - size);
     ctx.closePath();
     ctx.fill();
     ctx.globalAlpha = 1; // restore alpha
}    

function playPauseClick(){
  playingForward = true
     if(videoContainer !== undefined && videoContainer.ready){
          if(videoContainer.video.paused){                                 
                // videoContainer.video.play();
                document.querySelector("#playButton").innerHTML = "Pause"
                requestAnimationFrame(updateCanvas);
          }else{
                videoContainer.video.pause();
                document.querySelector("#playButton").innerHTML = "Play"
          }
     }
}

function playBackwards(){
  playingForward = false
  if(videoContainer !== undefined && videoContainer.ready){
       if(videoContainer.video.paused){                                 
             videoContainer.video.play();
             requestAnimationFrame(updateCanvas);
       }else{
            requestAnimationFrame(updateCanvas);
       }
} 
}

function nextFrame(){

  if(videoContainer !== undefined && videoContainer.ready){
    if(videoContainer.video.paused){
      console.log("Prueba frame")
      playNextFrame = true
      videoContainer.video.play()
      requestAnimationFrame(updateCanvas)
    }
  }
}

function getFrameNumberActual(){
  return calcularFrameActual(videoContainer.video)
}

function changeModo(){
  if (modo === modosAnotador.CORRECION){
    modo = modosAnotador.MODELO
  }else if(modo === modosAnotador.MODELO){
    modo = modosAnotador.CORRECION
  }
}

const handleCanvasEvents = evt =>{
  if (modo === modosAnotador.MODELO){

    puntos[currentFrame].push([evt.offsetX,evt.offsetY])
    console.log(puntos)
    drawPoints(ctx)
  }
}

function drawPoints(canvas){
  for(var i = 0; i < puntos[currentFrame].length; i++){
    console.log("drawing point...")
    x = puntos[currentFrame][i][0];
    y = puntos[currentFrame][i][1];

    canvas.beginPath();
    canvas.arc(x,y,5,0,2*Math.PI);
    canvas.stroke();
    canvas.closePath();
    canvas.fillStyle = "#0ce846";
    canvas.fill();
  }
}

function addLayerToFrame(canvas){
  if (puntos[currentFrame].length > 0){
    canvas.fillStyle = "black";
    canvas.beginPath();
    canvas.arc();
  }

}

function createPointData(){
  var frames = videoContainer.video.duration * fps;
  for(var i = 0; i < frames; i++){
    puntos[i] = new Array()
  }
  console.log(puntos)
  console.log(videoContainer.video.duration)
}
// function videoMute(){
//     muted = !muted;
// 	if(muted){
//          document.querySelector(".mute").textContent = "Mute";
//     }else{
//          document.querySelector(".mute").textContent= "Sound on";
//     }


// }
// register the event
canvas.addEventListener("click", handleCanvasEvents);

document.getElementById("playButton").addEventListener("click",playPauseClick);
document.getElementById("playButtonBackwards").addEventListener("click",playBackwards);
document.getElementById("clearButton").addEventListener("click",clearAllMasks);
document.getElementById("saveAnn").addEventListener("click",saveAnnotations);
// document.querySelector(".mute").addEventListener("click",videoMute)

document.getElementById("nextFrame").addEventListener("click",stepForward)
document.getElementById("lastFrame").addEventListener("click",stepBackward)
document.getElementById("changeModo").addEventListener("click",changeModo)

//
//
///////// Canvas event paint
//
//

const context = ctx;

const colour = "#3d34a5";
const strokeWidth = 8;

// Drawing state
let latestPoint;
let drawing = false;

const continueStroke = newPoint => {
  context.beginPath();
  context.moveTo(latestPoint[0], latestPoint[1]);
  context.strokeStyle = colour;
  context.lineWidth = strokeWidth;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.lineTo(newPoint[0], newPoint[1]);
  context.stroke();

  latestPoint = newPoint;
};

// Draw polygons
function dibujarPoligonos(poligonos) {
  // const canvas = document.getElementById('canvas');
  // const ctx = canvas.getContext('2d');

  // Limpiar el canvas antes de dibujar
  // ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Iterar sobre la lista de polígonos
  poligonos.forEach(poligono => {
      ctx.beginPath();

      // Moverse al primer punto del polígono
      ctx.moveTo(poligono[0][0], poligono[0][1]);

      // Dibujar líneas a los siguientes puntos
      for (let i = 1; i < poligono.length; i++) {
          ctx.lineTo(poligono[i][0], poligono[i][1]);
      }

      // Cerrar el polígono conectando el último punto con el primero
      ctx.closePath();

      // Estilo del polígono (relleno y borde)
      ctx.fillStyle = 'rgba(100, 150, 255, 0.5)'; // Relleno semitransparente
      ctx.fill();
      ctx.strokeStyle = 'blue';  // Color del borde
      ctx.lineWidth = 2;
      ctx.stroke();
  });
}

function addMasksToMainArray(masks,frameIndex=0){
  currentFrameAux = getFrameNumberActual()
  if (frameIndex != 0){
    if (!(frameIndex in all_masks)){
      all_masks[frameIndex] = new Array()
      // all_masks[currentFrameAux].push(Array())
      console.log("unidify")
    }
    all_masks[frameIndex].push(masks)
}
  else{
    if (!(currentFrameAux in all_masks)){
      all_masks[currentFrameAux] = new Array()
      // all_masks[currentFrameAux].push(Array())
      console.log("unidify")
    }
    all_masks[currentFrameAux].push(masks)
}
}


function drawMasks(){

  if (getFrameNumberActual() in all_masks){
    masks = all_masks[getFrameNumberActual()];
    console.log("dibujando");
    console.log(masks);
    masks.forEach(dibujarPoligonos);
  }



}

function addCorrectionToBackend(auxiliarCorrectionPoints){
  if (auxiliarCorrectionPoints.length > 0){
    maskIndex = checkClosestMask(auxiliarCorrectionPoints)
    addPointsToMaskAjaxRequest(maskIndex,auxiliarCorrectionPoints)
  }
  auxiliarCorrectionPoints = Array()

}

// const listaDePoligonos = [
//   // Triángulo
//   [[100, 100], [150, 50], [200, 100]],

//   // Cuadrado
//   [[300, 300], [400, 300], [400, 400], [300, 400]],

//   // Pentágono
//   [[500, 150], [550, 100], [600, 150], [575, 200], [525, 200]]
// ];


// Event helpers

auxiliarCorrectionPoints = Array()

const startStroke = point => {
  drawing = true;
  latestPoint = point;
};

const BUTTON = 0b01;
const mouseButtonIsDown = buttons => (BUTTON & buttons) === BUTTON;

// Event handlers

const mouseMove = evt => {
  if (!drawing) {
      return;
  }
  auxiliarCorrectionPoints.push([evt.offsetX, evt.offsetY]);
  continueStroke([evt.offsetX, evt.offsetY]);
  // console.log([evt.offsetX, evt.offsetY])
};

const mouseDown = evt => {
  if (modo === modosAnotador.CORRECION){
    if (drawing) {
      return;
  }
    evt.preventDefault();
    canvas.addEventListener("mousemove", mouseMove, false);
    auxiliarCorrectionPoints.push([evt.offsetX, evt.offsetY]);
    startStroke([evt.offsetX, evt.offsetY]);
    // console.log([evt.offsetX, evt.offsetY])
  }
  
};

function calculateDistance(point1, point2) {
  return Math.sqrt(Math.pow(point1[0] - point2[0], 2) + Math.pow(point1[1] - point2[1], 2));
}

function calculatePolygonDistance(polygon1, polygon2) {
  let minDistance = Infinity;
  console.log("Calculating distance")
  console.log(polygon1)
  console.log(polygon2)
  for (let i = 0; i < polygon1.length; i++) {
    for (let j = 0; j < polygon2.length; j++) {
      let distance = calculateDistance(polygon1[i], polygon2[j]);
      if (distance < minDistance) {
        minDistance = distance;
      }
    }
  }

  return minDistance;
}

// Create a javascript function so that given an array of masks and a polygon made by a list of points, checks which mask is the closest to the polygon
function checkClosestMask(polygon){
  console.log("Checking closest mask")
  console.log(polygon)

  let minDistance = Infinity
  let maskIndex = -1
  for (let i = 0; i < all_masks[getFrameNumberActual()].length; i++){
    let mask = all_masks[getFrameNumberActual()][i]
    let distance = calculatePolygonDistance(mask[0],polygon)
    console.log(distance);
    if (distance < minDistance){
      minDistance = distance
      maskIndex = i
    }
  return maskIndex
}
} 

const mouseEnter = evt => {
  if (!mouseButtonIsDown(evt.buttons) || drawing) {
      return;
  }
  mouseDown(evt);
};

const endStroke = evt => {
  if (!drawing) {
      return;
  }
  drawing = false;
  addCorrectionToBackend(auxiliarCorrectionPoints)
  evt.currentTarget.removeEventListener("mousemove", mouseMove, false);
};

// Register event handlers

canvas.addEventListener("mousedown", mouseDown, false);
canvas.addEventListener("mouseup", endStroke, false);
canvas.addEventListener("mouseout", endStroke, false);
canvas.addEventListener("mouseenter", mouseEnter, false);

function treat_masks(element){
  masks = Array()
  console.log(element)
  for (var j=0; j<element.length;j++){
    masks.push(Array())
    for(var i=0; i<element[j].length;i++){
      masks[j].push([element[j][i][0]*videoContainer.scale,(element[j][i][1]*videoContainer.scale)])
    }
  }

  return masks
}

$("#apply_sam").on('submit',function (e) {
  e.preventDefault();

  var videoRes = [videoContainer.video.videoWidth,videoContainer.video.videoHeight]
  currentFrame = calcularFrameActual(videoContainer.video)
  $.ajax({
  url:"/apply_sam",
  type:"POST",
  data: {puntos:JSON.stringify(puntos), current_frame:currentFrame, vPad:verticalPadding, vrW:videoRes[0], vrH:videoRes[1], crW:drawnVideoSize[0],crH:drawnVideoSize[1]},
  success: function(response){
      masks = treat_masks(response);
      addMasksToMainArray(masks)
      drawMasks();

  },
  error: function(error){
    console.log(error);
  },

  });

  return false;
});

$("#apply_cutie").on('submit',function (e) {
  e.preventDefault();
  // let puntos_ajax = puntos;
  var videoRes = [videoContainer.video.videoWidth,videoContainer.video.videoHeight]
  // var canvasRes = (canvas.height,canvas.width)


  // console.log("drawbdufbsdfisd")
  // console.log(drawnVideoSize)
  // console.log(videoRes)
  $.ajax({
  url:"/apply_cutie",
  type:"POST",
  data: {vPad:verticalPadding, vrW:videoRes[0], vrH:videoRes[1], crW:drawnVideoSize[0],crH:drawnVideoSize[1]},
  success: function(response){
      // masks = treat_masks(response);
      // // dibujarPoligonos(masks);
      // addMasksToMainArray(masks)
      masks_cutie = response
      console.log(masks_cutie)      
      //Substituir esta función por un desglose del dict y ejecutar treat masks para cada instancia dentro del dict
      //pensar en revisar que coinicidan los frames


      for(let key in masks_cutie){
        console.log("adding masks cutie")
        console.log(key)
        temp_masks = treat_masks(masks_cutie[key])
        console.log(temp_masks)
        addMasksToMainArray(masks_cutie[key],key)
      }
      drawMasks();

      // for(var i=0; i<masks.length;i++){
      //   temp_masks = treat_masks(masks[i])
      //   addMasksToMainArray(temp_masks,)
      // }
      // masks = treat_masks(response)
      // console.log(masks)
      // dibujarPoligonos(masks);

  },
  error: function(error){
    console.log(error);
  },

  });

  return false;
});

// generate a function that create an ajax request to the server that send a list of points to add to the currently selected mask
function addPointsToMaskAjaxRequest(maskIndex, points){
  console.log("Adding points to mask, backend request")
  console.log(maskIndex)
  console.log("maskIndex above, points below")
  console.log(points)
  var videoRes = [videoContainer.video.videoWidth,videoContainer.video.videoHeight]
  $.ajax({
    url:"/add_points_to_mask",
    type:"POST",
    data: {maskIndex:maskIndex, maskToAdd:JSON.stringify(points), vrW:videoRes[0], vrH:videoRes[1], crW:drawnVideoSize[0],crH:drawnVideoSize[1]},
    success: function(response){
      console.log(response)
      getMasksAjaxRequest()
    },
    error: function(error){
      console.log(error);
    },
  
    });
}

// generate a function that create an ajax request to the server that asks for all the masks to be sent to the frontend and then clean the current masks and add the new ones
function getMasksAjaxRequest(){
  var videoRes = [videoContainer.video.videoWidth,videoContainer.video.videoHeight]

  $.ajax({
    url:"/get_masks",
    type:"POST",
    data: {vPad:verticalPadding, vrW:videoRes[0], vrH:videoRes[1], crW:drawnVideoSize[0],crH:drawnVideoSize[1]},
    success: function(response){
      masks_cutie = response
      console.log(masks_cutie)      

      for(let key in masks_cutie){
        console.log("adding masks cutie")
        console.log(key)
        temp_masks = treat_masks(masks_cutie[key])
        console.log(temp_masks)
        addMasksToMainArray(masks_cutie[key],key)
      }
      drawMasks();

    },
    error: function(error){
      console.log(error);
    },
  
    });
}

function getFrames() {
  temporalFrames = Array()
  $.ajax({
    url:"/get_frames",
    type:"POST",
    data: {},
    success: function(response){
      masks_cutie = response
      let frames = masks_cutie.frames;
      console.log("frames") 
      console.log(frames)      

      frames.forEach((frame,index) => {
        let img = new Image();
        img.src = "data:image/jpeg;base64,"+frame;
        temporalFrames.push(img);
      });

      videoFrames = temporalFrames;
    },
    error: function(error){
      console.log(error);
    },
  
    });
}