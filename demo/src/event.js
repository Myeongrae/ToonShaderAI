import ndarray from "ndarray";
import ops from "ndarray-ops";
import * as ort from "onnxruntime-web";

// mouse movement on the light control pannel
let mouseOnLight = false;

function setLightCtrlPanel() {
    const glCanvasLight = document.querySelector("#light");

    glCanvasLight.addEventListener('mousedown', event => {
        console.log(`mousedown ${event}`);
        mouseOnLight = true;
    })
    
    glCanvasLight.addEventListener('mouseup', event => {
        console.log(`mouseup ${event}`);
        mouseOnLight = false;
    })
    
    glCanvasLight.addEventListener('mouseout', event => {
        console.log(`mouseout ${event}`);
        mouseOnLight = false;
    })
    
    glCanvasLight.addEventListener('mousemove', event=> {
        if (!mouseOnLight) {
            return;
        }
        console.log("mousemove");
        console.log(event);
        let degY = 0.05 * (event.movementX);
        let degX = 0.05 * (event.movementY);
    
        let cx = Math.cos(degX);
        let sx = Math.sin(degX);
        let cy = Math.cos(degY);
        let sy = Math.sin(degY);
    
        let [x, y, z] = light.position
        let x_ = cy*x + sx*sy*y + cx*sy*z;
        let y_ = cx*y - sx*z;
        let z_ = -sy*x + sx*cy*y + cx*cy*z;
    
        light.position = [x_, y_, z_];
        console.log(light.position);
    })
}


// light color picking
function hex2rgb(hex) {
    let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [parseInt(result[1], 16)/255, parseInt(result[2], 16)/255, parseInt(result[3], 16)/255] : null;
}

function rgb2hex(rgb) {
    function uint2hex(d) {
        let hex = d.toString(16);
        return hex.length == 1 ? "0" + hex : hex;
    }

    let [r, g, b] = rgb;
    let hex = "#" + uint2hex(Number(Math.round(r*255))) + uint2hex(Number(Math.round(g*255))) + uint2hex(Number(Math.round(b*255)));
    return hex;
}

function setColorPickerPanel() {
    const colorPickerLight = document.querySelector("#light_color");
    colorPickerLight.value = "#ffffff";

    colorPickerLight.addEventListener("change", (event) => {
        window.light.color = hex2rgb(event.target.value);
    });

    colorPickerLight.addEventListener("input", (event) => {
        window.light.color = hex2rgb(event.target.value);
    });

    const colorPickerAmbient = document.querySelector("#ambient_color");
    colorPickerAmbient.value = "#808080";

    colorPickerAmbient.addEventListener("change", (event) => {
        window.light.ambient = hex2rgb(event.target.value);
    });

    colorPickerAmbient.addEventListener("input", (event) => {
        window.light.ambient = hex2rgb(event.target.value);
    });
}

// style parameter control
function createStyleSliders() {
    const styleStdDiv = document.getElementById("style_std");

    for (let i = 0; i < 3; i++) {
        const div = document.createElement("div");
        styleStdDiv.appendChild(div);
        const slider = document.createElement("input");
        slider.setAttribute('type', 'range');
        slider.setAttribute('min', '0.0');
        slider.setAttribute('max', '100.0');
        slider.setAttribute('step', 'any');
        slider.setAttribute('value', '1.0');
        div.appendChild(slider);
        const text = document.createElement("input");
        text.setAttribute('type', 'number');
        text.setAttribute('min', '0.0');
        text.setAttribute('max', '100.0');
        text.setAttribute('value', '1.0');
        text.setAttribute('step', 'any');
        div.appendChild(text);
        slider.addEventListener("input", (event)=> {
            text.value = event.target.value;
            window.style.std[i] = event.target.value;
        })
    
        slider.addEventListener("change", (event)=> {
            text.value = event.target.value;
            window.style.std[i] = event.target.value;
        })
        text.addEventListener("change", (event) => {
            slider.value = event.target.value;
            window.style.std[i] = event.target.value;
        })
    }

    const styleMeanDiv = document.getElementById("style_mean");

    for (let i = 0; i < 3; i++) {
        const div = document.createElement("div");
        styleMeanDiv.appendChild(div);
        const slider = document.createElement("input");
        slider.setAttribute('type', 'range');
        slider.setAttribute('min', '-50.0');
        slider.setAttribute('max', '50.0');
        slider.setAttribute('step', 'any');
        slider.setAttribute('value', '1.0');
        div.appendChild(slider);
        const text = document.createElement("input");
        text.setAttribute('type', 'number');
        text.setAttribute('min', '-50.0');
        text.setAttribute('max', '50.0');
        text.setAttribute('value', '1.0');
        text.setAttribute('step', 'any');
        div.appendChild(text);
        slider.addEventListener("input", (event)=> {
            text.value = event.target.value;
            window.style.mean[i] = event.target.value;
        })
    
        slider.addEventListener("change", (event)=> {
            text.value = event.target.value;
            window.style.mean[i] = event.target.value;
        })
        text.addEventListener("change", (event) => {
            slider.value = event.target.value;
            window.style.mean[i] = event.target.value;
        })
    }
    
}

// run onnxruntime
function updateStyle() {
    for (let i = 0; i < 3; i++) {
        const divStd = document.getElementById("style_std").children[i]
        divStd.children[0].value = window.style.std[i];
        divStd.children[1].value = window.style.std[i];
        const divMean = document.getElementById("style_mean").children[i]
        divMean.children[0].value = window.style.mean[i];
        divMean.children[1].value = window.style.mean[i];
    }
    document.getElementById("light_color").value = rgb2hex(window.light.color);
    document.getElementById("ambient_color").value = rgb2hex(window.light.ambient);
};

function preProcess(ctx) {
    const imgData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const {data, width, height} = imgData;
    console.log('data preprocess', width, height);
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width*height*3), [1, 3, width, height]);
    ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 0));
    ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 2));
    ops.divseq(dataProcessedTensor, 255);

    const tensor = new ort.Tensor('float32', dataProcessedTensor.data, [1, 3, width, height]);
    return tensor;
};

function setTargetImageUploader() {
    const imgInput = document.querySelector("#file");
    const imgCanvas = document.querySelector("#imgCanvas");
    const ctx = imgCanvas.getContext("2d");

    imgInput.onchange = function(event){
        let img = new Image(128, 128);
        img.src = URL.createObjectURL(event.target.files[0]);
        img.onload = function() {
            console.log("image input size ", this.naturalWidth, this.naturalHeight);
            if (this.naturalWidth > this.naturalHeight) {
                ctx.drawImage(this, 
                    (this.naturalWidth - this.naturalHeight) / 2, 0, 
                    this.naturalHeight, this.naturalHeight, 
                    0, 0, 
                    this.width, this.height);
            }
            else {
                ctx.drawImage(this, 
                    0, (this.naturalHeight - this.naturalWidth) / 2, 
                    this.naturalWidth, this.naturalWidth, 
                    0, 0, 
                    this.width, this.height);
            }

            URL.revokeObjectURL(img.src);
            const tensor = preProcess(ctx);
            console.log(tensor);
            const results = assets.session.run({"input_0":tensor});
            results.then((value) => {
                window.style.mean = value["style_mean"].data;
                window.style.std = value["style_std"].data;
                window.light.color = value["light_color"].data;
                window.light.ambient = value["ambient_color"].data;
                updateStyle();
            })
        }
    };
}

export function prepareEvents() {
    setLightCtrlPanel();
    setColorPickerPanel();
    createStyleSliders();
    setTargetImageUploader();
}