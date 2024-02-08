import * as THREE from "three";
import * as ort from "onnxruntime-web";
import {FBXLoader} from "three/examples/jsm/loaders/FBXLoader.js"
import {OBJLoader} from "three/examples/jsm/loaders/OBJLoader.js"

window.assets = {};

window.style = {
    mean : [0.0, 0.0, 0.0],
    std : [1.0, 1.0, 1.0]
};

window.light = {
    position : [1.0, 0.0, 0.0],
    color : [1.0, 1.0, 1.0],
    ambient : [0.5, 0.5, 0.5] 
};

async function load3DObject(path) {
    let objLoader = new OBJLoader();
    if (path.slice(-3) == "fbx") {
        objLoader = new FBXLoader();
    } 

    const model = await objLoader.loadAsync(path);
    const array = {}

    model.traverse(o => {
        if(o instanceof THREE.Mesh) {
            console.log(o);
            for (const [key, value] of Object.entries(o.geometry.attributes)) {
                if (key == "position" || key == "normal" || key == "uv") {
                    array[key] = {numComponents : value.itemSize, data : value.array};
                }
            }
        }
    })
    return array;    
};

export async function loadAssets() {
    await Promise.all([
        (async () => {window.assets.session = await ort.InferenceSession.create('resources/model32.onnx', { executionProviders: ['webgl'] }); })(),
        (async () => {window.assets.objectBuffer = await load3DObject('resources/chocobunnyking.obj');})(),
        (async () => {window.assets.sphereBuffer = await load3DObject('resources/sphere_smooth.obj');})(),
    ])

    console.log("loading assets is done.")
};
