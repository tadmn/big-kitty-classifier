import gradio as gr
from fastai.vision.all import *

learn = load_learner('assets/models/learner.pkl')
labels = learn.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}





iface = gr.Interface(fn=predict, inputs=gr.Image(), outputs=gr.Label(num_top_classes=7), title='Big Kitty Classifier',
                     examples=['assets/examples/Tiger.jpeg', 'assets/examples/Lion.jpeg',
                               'assets/examples/Cheetah.jpeg', 'assets/examples/Panther.jpeg',
                               'assets/examples/Leopard.jpeg', 'assets/examples/Snow Leopard.jpeg',
                               'assets/examples/Cougar.jpeg'])

iface.launch(share=True)
