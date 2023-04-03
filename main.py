import os

from cvu.detector import Detector
import gradio as gr
import time

from flask import Flask

detector = Detector(classes="coco", backend="onnx")


def inference(image):
    start = time.time()
    preds = detector(image)
    delta = time.time() - start
    preds.draw(image)
    return (image, str(preds), f"FPS:{round(1/max(delta, 10e-9),2)}")


demo = gr.Interface(
    title="CVU Object Detection (Yolov5)",
    description=
    """<div style="background: #272822; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #f92672">from</span> <span style="color: #f8f8f2">cvu.detector</span> <span style="color: #f92672">import</span> <span style="color: #f8f8f2">Detector</span>
<span style="color: #f8f8f2">detector</span> <span style="color: #f92672">=</span> <span style="color: #f8f8f2">Detector(classes</span><span style="color: #f92672">=</span><span style="color: #e6db74">&quot;coco&quot;</span><span style="color: #f8f8f2">,</span> <span style="color: #f8f8f2">backend</span><span style="color: #f92672">=</span><span style="color: #e6db74">&quot;onnx&quot;</span><span style="color: #f8f8f2">)</span>
<span style="color: #f8f8f2">preds</span> <span style="color: #f92672">=</span> <span style="color: #f8f8f2">detector(image)</span>
<span style="color: #f8f8f2">preds</span><span style="color: #f92672">.</span><span style="color: #f8f8f2">draw(image),</span> <span style="color: #66d9ef">print</span><span style="color: #f8f8f2">(str(preds))</span>
</pre></div>
""",
    fn=inference,
    inputs=["image"],
    outputs=["image", "text", "text"],
)
demo.launch()

# gr.Interface()

# gr.Image(source="webcam", streaming=True)

app = Flask(__name__)
# io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == '__main__':
    app.run(host=0.0.0.0, port=os.getenv("PORT", default=5000))
