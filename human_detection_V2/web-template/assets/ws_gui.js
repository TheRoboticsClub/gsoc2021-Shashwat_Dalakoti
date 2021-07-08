// Set the src of iframe tag

// To decode the image string we will receive from server
function decode_utf8(s) {
    return decodeURIComponent(escape(s))
}

// Websocket and other variables for image display
var websocket_gui;

function declare_gui() {
    websocket_gui = new WebSocket("ws://" + websocket_address + ":2303/");

    websocket_gui.onopen = function (event) {
        if (websocket_code.readyState == 1)
            alert("[open] Connection established!");
    }

    websocket_gui.onclose = function (event) {
        radiConect.contentWindow.postMessage('down', '*');
        if(event.wasClean){
            alert(`[close] Connection closed cleanly, code=${event.code} reason=${event.reason}`);
        }
        else{
            alert("[close] Connection closed!");
        }
    }

    // What to do when a message from server is received
    websocket_gui.onmessage = function (event) {
        var operation = event.data.substring(0, 4);
        radiConect.contentWindow.postMessage('up', '*');
        if (operation == "#gui") {
            // Parse the entire Object
            var data = JSON.parse(event.data.substring(4,));

            // Parse the Image Data
            var image_data = JSON.parse(data.image),
                source = decode_utf8(image_data.image),
                count = image_data.count;
                shape = image_data.shape;

            if (source != "") {
                image.src = "data:image/jpeg;base64," + source;
                canvas.width = shape[1];
                canvas.height = shape[0];
            }

            if (count != ""){
                if (count == "Plot"){
                    document.getElementById("output_heading").textContent = "Model Benchmarking Results"
                }
                else{
                    document.getElementById("output_heading").textContent = "Total People in Frame: " + count
                }
            }

            // Parse the Console messages
            messages = JSON.parse(data.text_buffer);
            // Loop through the messages and print them on the console
            for (message of messages) {
                // Set value of command
                command.value = message
                // Go to next command line
                next_command()
            }
            // Send the Acknowledgment Message
            websocket_gui.send("#ack");

        } else if (operation == "#cor") {
            // Set the value of command
            var command_input = event.data.substring(4,);
            command.value = command_input;
            // Go to next command line
            next_command();
            // Focus on the next line
            command.focus();
        }


    }
}

var canvas = document.getElementById("gui_canvas"),
    context = canvas.getContext('2d');
    image = new Image();

// For image object
image.onload = function () {
    update_image();
}

// Request Animation Frame to remove the flickers
function update_image() {
    window.requestAnimationFrame(update_image);
    context.drawImage(image, 0, 0);
}

