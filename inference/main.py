import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from config import InferenceConfig
from inference import LlamaInterface, Qwen2VLInterface, InternVLInterface

def main(cli:bool,engine):
    
    ### test image
    imga_path = '/root/Documents/project/qwenvl_infer/demo.jpeg'
    from PIL import Image
    # load image
    image = Image.open(imga_path)
    flag = 0
    if cli:
        while True:
            line = input()
            if flag == 0:
                print(engine.predict(line, image))
                flag = 1
            else:
                print(engine.predict(line))
    from flask import Flask, request, jsonify
    from flask import render_template  # 引入模板插件
    from flask_cors import CORS
    pool = ThreadPoolExecutor(max_workers=2)        
    app = Flask(
        __name__,
        static_folder='./dist',  # 设置静态文件夹目录
        template_folder="./dist",
        static_url_path=""
    )

    CORS(app, resources=r'/*')
    
    @app.route('/')
    def index():
        return render_template('index.html', name='index')

    @app.route("/api/chat", methods=["POST"])
    def getChat():
        msg = request.get_json(force=True)['message']
        if len(msg) == 0:
            return jsonify({"code": 404})
        pool.submit(engine.predict,msg)
        return jsonify({"code": 200})

    @app.route("/api/getMsg", methods=["GET"])
    def getMsg():
        return jsonify(engine.getState())
    
    @app.route("/api/reset", methods=["GET"])
    def reset():
        engine.reset()
        return jsonify({"code": 200})

    app.run(
        use_reloader=False,
        host="0.0.0.0",
        port=5000
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cli', dest='cli', default=False, action='store_true',
        help="run web ui by default, if add --cli, run cli."
    )
    parser.add_argument("--kv_size", type=int, default=256)
    parser.add_argument(
        "--engine", type=str, default="acl",
        help="inference backend, onnx or acl"
    )
    parser.add_argument(
        "--sampling", type=str, default="top_k",
        help="sampling method, greedy, top_k or top_p"
    )
    parser.add_argument(
        "--sampling_value",type=float,default=10,
        help="if sampling method is seted to greedy, this argument will be ignored; if top_k, it means value of p; if top_p, it means value of p"
    )
    parser.add_argument(
        "--temperature",type=float,default=0.7,
        help="sampling temperature if sampling method is seted to greedy, this argument will be ignored."
    )
    parser.add_argument(
        "--hf-dir", type=str, default="/root/model/tiny-llama-1.1B", 
        help="path to huggingface model dir"
    )
    parser.add_argument(
        "--model", type=str, default=None, 
        help="path to onnx or om model"
    )
    parser.add_argument(
        "--vision_model", type=str, default=None,
        help="path to vision model"
    )
    parser.add_argument(
        "--embedding_model", type=str, default=None,
        help="path to embedding model"
    )
    parser.add_argument(
        "--llm_model", type=str, default=None,
        help="path to llm model"
    )
    parser.add_argument(
        "--mlp_model", type=str, default=None,
        help="path to mlp model"
    )
    parser.add_argument(
        "--model_type", type=str, default=None,
        help="choose the type of model, qwen2vl-2b, llama-2-7b, or..."
    )
    parser.add_argument(
        "--kvcache", type=str, default="basic",
        help="choose the type of kvcahe, 'basic'|'sliding-window'|'streamllm'|'H2O'"
    )
    parser.add_argument(
        "--visual_path", type=str, default=None,
        help="path to images for visual inference"
    )
    parser.add_argument(
        "--pact_config_path", type=str, default=None,
        help="path to pact config file"
    )
    parser.add_argument(
        "--tokenbytoken", type=bool, default=False,
        help="if set to True, the model will generate token by token output"
    )
    args = parser.parse_args()
    cfg = InferenceConfig(
        hf_model_dir=args.hf_dir,
        model=args.model,
        vision_model=args.vision_model,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        mlp_model=args.mlp_model,
        sampling_method=args.sampling,
        sampling_value=args.sampling_value,
        session_type=args.engine,
        kvcache_method=args.kvcache,
        max_cache_size=args.kv_size,
        model_type=args.model_type,
        visual_path=args.visual_path,
        pact_config_path=args.pact_config_path,
        is_token_by_token=args.tokenbytoken,
    )

    if args.model_type == "qwen2vl-2b" or args.model_type == "qwen2vl-pact":
        engine = Qwen2VLInterface(cfg)
    elif args.model_type == "internvl":
        engine = InternVLInterface(cfg)
    elif args.model_type == "llama-2-7b":
        engine = LlamaInterface(cfg)
    else:
        print("Invalid model type")
        sys.exit(1)
    main(args.cli,engine)