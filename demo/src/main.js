"use strict";

import * as twgl from "twgl.js";

import { vsSource, fsSource, fsSphere } from "./shader.js";
import { loadAssets } from "./preload.js";
import { prepareEvents } from "./event.js";

main();

window.onload = (event) => {
    prepareEvents();
}

// main function
async function main() {
    
    console.log('Hellow world?');

    await loadAssets();
    console.log(window.assets);

    // webGL Initialize
    const glCanvasToon = document.querySelector("#toon");
    const glCanvasLight = document.querySelector("#light");
    const glParentCanvas = glCanvasToon.parentElement;
    const checkOcclusion = document.querySelector("#occlusion");
    const checkEmission = document.querySelector("#emission");
    const checkShininess = document.querySelector("#shininess");

    const canvas = document.querySelector("#glcanvas");
    const gl = canvas.getContext("webgl2");
    
    if(!gl) {
        alert(
            "Unable to initialize WebGL. Your browser or machine may not support it.",
        );
        return;
    }

    twgl.setDefaults({attribPrefix: 'a_'});
    
    // load shaders
    const programInfo = twgl.createProgramInfo(gl, [vsSource, fsSource]);
    const sphereProgramInfo = twgl.createProgramInfo(gl, [vsSource, fsSphere]);

    // load textures
    const textures = twgl.createTextures(gl, {
        albedo : {src : "resources/chocobunnyking_d.png"},
        occlusion : {src : "resources/chocobunnyking_o.png"},
        emission : {src : "resources/chocobunnyking_e.png"},
        shininess : {src : "resources/chocobunnyking_s.png"},
        white : {src : "resources/white.png"},
        black : {src : "resources/black.png"}
    });

    // create a buffer for each array 
    const bufferInfo = twgl.createBufferInfoFromArrays(gl, window.assets.objectBuffer);
    const sphereBufferInfo = twgl.createBufferInfoFromArrays(gl, window.assets.sphereBuffer);

    // create VAO and recording
    const vao = twgl.createVAOFromBufferInfo(gl, programInfo, bufferInfo);
    const sphereVAO = twgl.createVAOFromBufferInfo(gl, sphereProgramInfo, sphereBufferInfo);

    // prepare uniform variables 
    const cameraTarget = [0, 0, 0];
    const cameraPosition = [0, 0, 3];
    const zNear = 0.1;
    const zFar = 50;

    function degToRad(deg) {
        return deg*Math.PI / 180;
    }

    const fieldOfViewRadians = degToRad(60);
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const projection = twgl.m4.perspective(fieldOfViewRadians, aspect, zNear, zFar);

    const up = [0, 1, 0];
    const camera = twgl.m4.lookAt(cameraPosition, cameraTarget, up);
    const view = twgl.m4.inverse(camera);

    const uniforms = {
        u_view : view,
        u_projection : projection,
        u_world : twgl.m4.identity(),
        u_albedo : textures["albedo"],
        u_occlusion : textures["occlusion"],
        u_emission : textures["emission"],
        u_shininess : textures["shininess"],
        u_lightPos : window.light.position,
        u_lightColor : window.light.color,
        u_ambientColor : window.light.ambient,
        u_styleMean : window.style.mean,
        u_styleStd : window.style.std
    };

    // create drawing tasks
    const tasks = [
        {
            programInfo: programInfo,
            bufferInfo: bufferInfo,
            vertexArray: vao,
            uniforms: uniforms,
            element: glCanvasToon
        },

        {
            programInfo: sphereProgramInfo,
            bufferInfo: sphereBufferInfo,
            vertexArray: sphereVAO,
            uniforms: uniforms,
            element: glCanvasLight
        }
    ]

    function drawScene(task) {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(task.programInfo.program);

        gl.bindVertexArray(task.vertexArray);

        twgl.setUniforms(task.programInfo, task.uniforms);

        twgl.drawBufferInfo(gl, task.bufferInfo);
    }

    function render(time) {
        time *= 0.001; // converts to seconds;
        twgl.resizeCanvasToDisplaySize(gl.canvas);
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
        gl.enable(gl.SCISSOR_TEST);

        tasks.forEach(task => {
            const rect = task.element.getBoundingClientRect();
            const glRect = glParentCanvas.getBoundingClientRect();

            const width = rect.width;
            const height = rect.height;
            const left = rect.left - glRect.left;
            const bottom = gl.canvas.clientHeight - rect.bottom + glRect.top;

            // if(bottom < 0 || bottom + height > gl.canvas.clientHeight ||
            //     left < 0 || left + width > gl.canvas.clientWidth) {
            //     return;
            // }

            gl.viewport(left, bottom, width, height);
            gl.scissor(left, bottom, width, height);
            task.uniforms.u_world = twgl.m4.rotationY(time);
            task.uniforms.u_styleMean = style.mean;
            task.uniforms.u_styleStd = style.std;
            task.uniforms.u_lightColor = light.color;
            task.uniforms.u_ambientColor = light.ambient;
            task.uniforms.u_lightPos = light.position;

            task.uniforms.u_occlusion = checkOcclusion.checked? textures["occlusion"] : textures["white"];
            task.uniforms.u_emission = checkEmission.checked? textures["emission"] : textures["black"];
            task.uniforms.u_shininess = checkShininess.checked? textures["shininess"] : textures["black"];
            

            drawScene(task);
        })

        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);

}

