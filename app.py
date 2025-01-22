import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load and process image
def load_and_process_image(image, max_dim=512):
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img * 255.0

# Deprocess image
def deprocess_image(img):
    img = img[0]
    img = tf.clip_by_value(img, 0, 255)
    img = tf.cast(img, tf.uint8)
    return img

# Display image
def show_image(image, title):
    image = image[0]
    plt.imshow(tf.cast(image, tf.float32) / 255.0)
    if title:
        plt.title(title)
    plt.show()

# VGG19 model
def get_vgg_model(style_layers, content_layer):
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_output = vgg.get_layer(content_layer).output
    model_outputs = style_outputs + [content_output]
    return Model(vgg.input, model_outputs)

# Loss functions
def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def total_variation_loss(image):
    x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    return tf.reduce_mean(tf.square(x_deltas)) + tf.reduce_mean(tf.square(y_deltas))

def compute_loss(model, init_image, gram_style_features, content_features, style_weight=1e-2, content_weight=1e4):
    model_outputs = model(init_image)
    style_outputs = model_outputs[:num_style_layers]
    content_output = model_outputs[num_style_layers:]

    style_score = 0
    for target_style, comb_style in zip(gram_style_features, style_outputs):
        style_score += style_loss(comb_style, target_style)
    style_score *= style_weight / num_style_layers

    content_score = content_loss(content_output[0], content_features[0])
    content_score *= content_weight

    total_loss = style_score + content_score
    return total_loss

# Streamlit UI
st.title("Neural Style Transfer")

content_image = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_image and style_image:
    content_image = content_image.read()
    style_image = style_image.read()

    content_image = load_and_process_image(content_image)
    style_image = load_and_process_image(style_image)

    st.image(deprocess_image(content_image).numpy(), caption="Content Image", use_column_width=True)
    st.image(deprocess_image(style_image).numpy(), caption="Style Image", use_column_width=True)

    if st.button("Start Style Transfer"):
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'
        num_style_layers = len(style_layers)

        model = get_vgg_model(style_layers, content_layer)

        def get_feature_representations(model, content_image, style_image):
            style_outputs = model(style_image)
            content_output = model(content_image)

            style_features = [style_layer for style_layer in style_outputs[:num_style_layers]]
            content_features = [content_output[num_style_layers]]

            return style_features, content_features

        style_features, content_features = get_feature_representations(model, content_image, style_image)
        gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

        init_image = tf.Variable(content_image, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=0.02)

        @tf.function()
        def train_step(init_image, model, gram_style_features, content_features, optimizer):
            with tf.GradientTape() as tape:
                loss = compute_loss(model, init_image, gram_style_features, content_features)
                loss += total_variation_loss(init_image)

            grad = tape.gradient(loss, init_image)
            optimizer.apply_gradients([(grad, init_image)])
            init_image.assign(tf.clip_by_value(init_image, 0.0, 255.0))

            return loss

        epochs = 20000
        for epoch in range(epochs):
            loss = train_step(init_image, model, gram_style_features, content_features, optimizer)
            if epoch % 100 == 0:
                st.write(f"Epoch {epoch}/{epochs} - Loss: {loss.numpy()}")
                st.image(deprocess_image(init_image).numpy(), caption=f"Step {epoch}", use_column_width=True)

        st.image(deprocess_image(init_image).numpy(), caption="Output Image", use_column_width=True)