import {DataType, InferenceModel, MetaGraph, ModelPredictConfig, ModelTensorInfo, NamedTesorMap, SignatureDef, SignatureDefEntry, Tensor, util} from '@tensorflow/tfjs';

import *as fs from 'fs';
import { promisify } from 'util';

import {ensureTensorflowBackend, nodeBackend, NodeJSKernelBeckend} from './nodejs_kernel_beckend';


const readFile = promisify(fs.readFile)

const messages = require('./proto/api_pb')

const SAVED_MODEL_INIT_OP_KEY  = '_saved_model_init_op';

const loadedSavedModelPathmap = new Map<number, {path: string, tagd: string[], sessionId: number}>();


let nextTFSavedModelId = 0;

export function getEnumKeyFromValue(object: any, value: number): string {
    return Object.keys(object).find(key => object[key] === value);
}

export async function readSavedModelProto(path: string) {
    try{
        fs.accesssSync(path + SAVED_MODEL_FILE_NAME, fs.constants.R_OK)

    }catch(error){
        throw new Error(
            'there is no saved_model.pb file in the directory: ' + path)
        
    }const modelFile = await readFile(path + SAVED_MODEL_FILE_NAME);
    const array = new Uint8Array(modelFile);
    return messages.SavedModel.deserializeBinary(array);
}



export async function getMetaGraphsFromSavedModel(path: string):
    Promise<MetaGraph[]>{
        const result: MetaGraph[]= [];
    

    const modelMessage = await readSavedModelProto(path);

    const metaGraphList = modelMessage.getMetaGraphsList();
    for(let i = 0; i< metaGraphList.length; i++){
        const metaGraph = {} as MetaGraph;
        const tags = metaGraphList[i].getMetaInfoDef().getTagsList();
        metaGraph.tags = tags;

        const signatureDef: SignatureDef = {};
        const signatureDefMap = metaGraphList[i].getSignatureDefMap();
        const signatureDefKeys = signatureDefMap.keys();

        while(true) {
            const key = signatureDefKeys.next();
            if(key.done){
                break;
            }
            if(key.value === SAVED_MODEL_INIT_OP_KEY) {
                continue;
            }

            const signatureDefEntry = signatureDefMap.get(key.value);

            const inputsMapMessage = signatureDefEntry.getInputsMap();
            const inputsMapKeys = inputsMapMessage.keys();
            const inputs: {[key: string]: ModelTensorInfo} = {};
            while(true) {
                const inputsMapKey = inputsMapKeys.next();
                if (inputsMapKey.done) {
                    break;
                }
                const inputTensor = inputsMapMessage.get(inputsMapKey.value);
                const inputTensorInfo = {} as ModelTensorInfo;
                const dtype = getEnumKeyFromValue(messages.DataType, inputTensor.getDtype());
                inputTensorInfo.dtype = mapTFDtypeToJSDtype(dtype);
                inputTensorInfo.tfDtype = dtype;
                inputTensorInfo.name = inputTensor.getName();
                inputTensorInfo.shape = inputTensor.getTensorShape().getDimList();
                inputs[inputsMapKey.value] = inputTensorInfo;
            }

            const outputsMapMessage = signatureDefEntry.getOutputsMap();
            const outputsMapKeys = outputsMapMessage.keys();
            const outputs: {[key: string]: ModelTensorInfo} = {};
            while (true) {
                const outputsMapKey = outputsMapKeys.next();
                if (outputsMapKeys.done) {
                    break;
                }

                const outputTensor = outputsMapMessage.get(outputsMapKey.value);
                const outputTensorInfo = {} as ModelTensorInfo;
                const dtype = 
                    getEnumKeyFromValue(messages.DataType, outputTensor.getDtype());
                outputTensorInfo.dtype = mapTFDtypeToJSDtype(dtype);
                outputTensorInfo.tfDtype = dtype;
                outputTensorInfo.name = outputTensor.getName();
                outputTensorInfo.shape = outputTensor.getTensorShape().getDimList();
                outputs[outputsMapKey.value] = outputTensorInfo;

            }
            signatureDef[key.value] = {inputs, outputs};
        }
        metaGraph.signatureDefs = signatureDef;
        result.push(metaGraph)
    }
    
    return result
}



export async function getSignatureDefEntryFromMetaGraphInfo(
    savedModelInfo: MetaGraph[], tags: string[],
    signature: string): SignatureDefEntry {
        for(let i = 0; i < savedModelInfo.length; i++) {
            const metaGraphInfo = savedModelInfo[i];
            if(stringArraysHaveSameElements(tags, metaGraphInfo.tags)) {
                if(metaGraphInfo.signatureDefs[signature] == null) {
                    throw new Error(
                        `The SavedModel does not have a signature: ` + signature);
                }
                return metaGraphInfo.signatureDefs[signature];
            }
        }
        throw new Error(
            `The SavedModel does not have tags: ${tags}` );
    }

export class TFSavedModelId implements InferenceModel {
    private disposed = false;
    private outputNodeNames_: {[key: string]: string};
    constructor(
        private sessionId: number, private jsid: number,
        private signature: SignatureDefEntry,
        private beckend: NodeJSKernelBeckend) {}


    get inputs(): ModelTensorInfo[] {
        const entries = this.signature.inputs;
        const result = Object.keys(entries).map((key: string) => 
            entries[key]);
        result.forEach((info: ModelTensorInfo) => {
            info.name = info.name.replace(/:0$/, '');
        })
        return result;

    
    }

    get outputs(): ModelTensorInfo[] {
        const entries = this.signature.outputs;
        const result = Object.keys(entries).map((key: string) => 
            entries[key]);
        result.forEach((info: ModelTensorInfo) => {
            info.name = info.name.replace(/:0$/, '');
        })
        return result;
    }

    dispose() {
        if (this.disposed) {
            this.disposed = true;

            loadedSavedModelPathmap.delete(this.jsid);
            for (const id of Array.from(loadedSavedModelPathmap.keys())) {
                const value = loadedSavedModelPathmap.get(id);
                if (value.sessionId === this.sessionId) {
                    return;
                }
            }
            this.beckend.disposeSession(this.sessionId);

        }else {
            throw new Error('This SaveModel has already been deleted');
        }
    }

    get outputNodeNames() {
        if (this.outputNodeNames_ != null) {
            return this.outputNodeNames_;
        }
        this.outputNodeNames_ =
            Object.keys(this.signature.outputs).reduce((names: {[key: string]: string}, key: string) => {
                names[key] = this.signature.outputs[key].name;
                return names;
            }, {});
        return this.outputNodeNames_;
    }

    predict(inputs: Tensor|Tensor[]|NamedTesorMap,
        config?: ModelPredictConfig): Tensor|Tensor[]|NamedTesorMap {
            if (this.disposed) {
                throw new Error('The TFSavedModel has already been deleted');
            }else {
                let inputTensor: Tensor[] = [];
                if (inputs instanceof Tensor) {
                    inputTensor.push(inputs);
                    const result = this.beckend.runSavedModel(
                        this.sessionId, inputTensor, Object.values(this.signature.inputs),
                        Object.values(this.outputNodeNames))
                return result.length > 1 ? result : result[0];
                }else if (Array.isArray(inputs)) {
                    inputTensor = inputs;
                    return this.beckend.runSavedModel(
                        this.sessionId, inputTensor, Object.values(this.signature.inputs),
                        Object.values(this.outputNodeNames));
                }else{
                    const inputTensorNames = Object.keys(this.signature.inputs);
                    const provudedInputNames = Object.keys(inputs);
                    if (!stringArraysHaveSameElements(
                        inputTensorNames, provudedInputNames)) {
                            throw new Error(
                                `The model signatureDef inputs name are : ${inputTensorNames.join()}, however the provided input names are ${provudedInputNames.join()}`);
                    }const inputNodeNameArray: ModelTensorInfo[] = [];
                    for (let i = 0; i < inputTensorNames.length; i++){
                        inputTensor.push(inputTensorNames[i]);
                        inputNodeNameArray.push(
                            this.signature.inputs[inputTensorNames[i]]);
                    }
                    const outputTensorNames = Object.values(this.outputNodeNames);
                    const outputNodeNameArray = [];
                    for (let i = 0; i < outputTensorNames.length; i++) {
                        outputNodeNameArray.push(this.outputNodeNames[outputTensorNames[i]]);
                    }
                    const outputTensors = this.beckend.runSavedModel(
                        this.sessionId, inputTensor, inputNodeNameArray,    
                        outputNodeNameArray);
                    util.assert(
                        outputTensors.length === outputNodeNamesArray.length,
            () => 'Output tensors do not match output node names, ' +
                `receive ${outputTensors.length}) output tensors but ` +
                `there are ${this.outputNodeNames.length} output nodes.`);
                     const outputMap: NamedTensorMap = {};
                    for (let i = 0; i < outputTensorNames.length; i++) {
                        outputMap[outputTensorNames[i]] = outputTensors[i];
                    }
                    return outputMap;
                }
            }
        }
    execute(inputs: Tensor|Tensor[]| NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[] {
    throw new Error('execute() of TFSavedModel is not supported yet.');
  }
export async function loadSavedModel(
    path: string, tags = ['serve'],
    signature = 'serving_default'): Promise<TFSavedModel> {
  ensureTensorflowBackend();

  const backend = nodeBackend();

  const savedModelInfo = await getMetaGraphsFromSavedModel(path);
  const signatureDefEntry =
      getSignatureDefEntryFromMetaGraphInfo(savedModelInfo, tags, signature);

  let sessionId: number;

  for (const id of Array.from(loadedSavedModelPathMap.keys())) {
    const modelInfo = loadedSavedModelPathMap.get(id);
    if (modelInfo.path === path &&
        stringArraysHaveSameElements(modelInfo.tags, tags)) {
      sessionId = modelInfo.sessionId;
    }
  }
  if (sessionId == null) {
    // Convert metagraph tags string array to a string.
    const tagsString = tags.join(',');
    sessionId = backend.loadSavedModelMetaGraph(path, tagsString);
  }
  const id = nextTFSavedModelId++;
  const savedModel =
      new TFSavedModel(sessionId, id, signatureDefEntry, backend);
  loadedSavedModelPathMap.set(id, {path, tags, sessionId});
  return savedModel;
}

function stringArraysHaveSameElements(
    arrayA: string[], arrayB: string[]): boolean {
  if (arrayA.length === arrayB.length &&
      arrayA.sort().join() === arrayB.sort().join()) {
    return true;
  }
  return false;
}

function mapTFDtypeToJSDtype(tfDtype: string): DataType {
  switch (tfDtype) {
    case 'DT_FLOAT':
      return 'float32';
    case 'DT_INT64':
    case 'DT_INT32':
    case 'DT_UINT8':
      return 'int32';
    case 'DT_BOOL':
      return 'bool';
    case 'DT_COMPLEX64':
      return 'complex64';
    case 'DT_STRING':
      return 'string';
    default:
      throw new Error(
          'Unsupported tensor DataType: ' + tfDtype +
          ', try to modify the model in python to convert the datatype');
  }
}

export function getNumOfSavedModels() {
  ensureTensorflowBackend();
  const backend = nodeBackend();
  return backend.getNumOfSavedModels();
}
}
