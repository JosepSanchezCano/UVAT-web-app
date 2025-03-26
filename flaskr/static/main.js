
document.getElementById("upload-file").onchange = function(){
    $.post("/load_video", {
        javascript_data: $("#upload_file").val()
    });
};

// $.ajax({
//     type:"POST",
//     url:"http://ruta",
//     data:{}
// })