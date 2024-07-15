
document.getElementById("upload-file").onchange = function(){
    $.post("/load_video", {
        javascript_data: $("#upload_file").val()
    });
};
