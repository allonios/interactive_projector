"""Compute depth maps for images in the input folder.
"""
import cv2
import numpy as np
import onnxruntime as rt

from utils.monocular_depth.midas.transforms import PrepareForNet, Resize


def load_model(model_path, model_type="small"):
    if model_type == "large":
        net_w, net_h = 384, 384
    elif model_type == "small":
        net_w, net_h = 256, 256
    else:
        print(f"model_type '{model_type}' not implemented use small or larg")
        assert False

    print("loading model...")
    model = rt.InferenceSession(model_path)
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    resize_image = Resize(
        net_w,
        net_h,
        resize_target=None,
        keep_aspect_ratio=False,
        ensure_multiple_of=32,
        resize_method="upper_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    )

    def compose2(f1, f2):
        return lambda x: f2(f1(x))

    transform = compose2(resize_image, PrepareForNet())

    return model, transform, input_name, output_name, net_w, net_h


model, transform, input_name, output_name, net_w, net_h = load_model(
    "midas/model-small.onnx"
)


def find_depth(
    image,
):
    image_input = transform({"image": image})["image"]

    # compute
    output = model.run(
        [output_name],
        {
            input_name: image_input.reshape(1, 3, net_h, net_w).astype(
                np.float32
            )
        },
    )[0]
    prediction = np.array(output).reshape(net_h, net_w)
    prediction = cv2.resize(
        prediction,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    return prediction


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("failed to read image")

    cv2.imshow("Image", image)

    depth = find_depth(image)

    center_depth = depth[int(depth.shape[1] / 2), int(depth.shape[0] / 2)]
    print(center_depth)

    if cv2.waitKey(1) & 0xFF == 27:
        exit()
