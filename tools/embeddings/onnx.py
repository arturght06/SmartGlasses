import onnxruntime

# Создание объекта SessionOptions
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

print(onnxruntime.get_available_providers())


# Создание сессии с явным указанием провайдера CUDA
session = onnxruntime.InferenceSession(
    "other/arcfaceresnet100-8.onnx",
    sess_options=session_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("Используемый Execution Provider:", session.get_providers())
