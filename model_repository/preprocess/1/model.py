import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            img = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()
            x = (img.astype(np.float32) - 127.5) / 128.0
            x = x.transpose(2, 0, 1)[None].astype(np.float32)
            out = pb_utils.Tensor("PREPROCESSED", x)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
