import numpy as np
from ais_bench.infer.interface import InferSession

def infer_api_static():
    device_id =0
    model_path="/root/Documents/model_dir/Ascend_llms/Internvl/internvl_visual_model_test_3.om"
    # create session of om model for inference
    session=InferSession(device_id,model_path)
    print(f"session initialized.")
    #create new numpy data according inputs info
    shape0 =session.get_inputs()[0].shape
    print(f"shape0:{shape0}")
    ndata0 =np.full(shape0,1).astype(np.float16)
    feeds = [ndata0]
    # execute inference, inputs is ndarray list and outputs is ndarray listoutputs=session.infer(feeds,mode='static')
    outputs=session.infer(feeds,mode='static')
    print(f"outputs:{outputs[0].shape}")
    print(f"outputs:{outputs}")
    # free model resource and device context of session
    session.free_resource()

infer_api_static()