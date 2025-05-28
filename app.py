import streamlit as st
import numpy as np
import cv2
from PIL import Image, Image as PILImage
from skimage import exposure, transform
import io

st.set_page_config(page_title="Preprocesamiento de Imágenes", layout="wide")
st.title("🧠 Procesamiento Automático de Imágenes Subidas")

uploaded_files = st.file_uploader(
    "📁 Sube una o más imágenes", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

def load_image(file):
    img = Image.open(file).convert("RGB")
    return np.array(img)

def resize_image(img, scale):
    resized = transform.rescale(img, scale, channel_axis=-1, mode='reflect', preserve_range=True)
    return resized.astype(np.uint8)

def equalize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    eq = exposure.equalize_hist(gray)
    eq_rgb = np.stack((eq,) * 3, axis=-1)
    return (eq_rgb * 255).astype(np.uint8)

def apply_linear_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 25
    filtered = cv2.filter2D(gray, -1, kernel)
    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

def apply_nonlinear_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filtered = cv2.medianBlur(gray, 5)
    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

def translate_image(img):
    rows, cols = img.shape[:2]
    dx, dy = cols // 2, rows // 2  # Mover a mitad del canvas
    canvas_width, canvas_height = cols * 2, rows * 2  # duplicar lienzo

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(img, M, (canvas_width, canvas_height), borderValue=(0, 0, 0))
    return translated

def rotate_image(img):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows), borderValue=(0, 0, 0))
    return rotated

def mostrar_imagen(imagen_np, caption="", ancho=None):
    img_pil = PILImage.fromarray(imagen_np)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=caption, width=ancho if ancho else None)

# Procesar imágenes
if uploaded_files:
    for file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📸 Imagen: {file.name}")
        img = load_image(file)

        col1, col2, col3 = st.columns([1, 0.1, 2])
        # Ancho fijo base para original y transformaciones iguales
        ANCHO_BASE = 300

        with col1:
            mostrar_imagen(img, caption=f"🖼️ Original - {img.shape[1]}x{img.shape[0]}", ancho=ANCHO_BASE)

        with col3:
            # AGRANDADO (más grande)
            img_big = resize_image(img, 2.0)
            mostrar_imagen(img_big, caption=f"🔍 Reescalado (Agrandado) - {img_big.shape[1]}x{img_big.shape[0]}", ancho=ANCHO_BASE * 2)

            # REDUCIDO (más pequeño)
            img_small = resize_image(img, 0.5)
            mostrar_imagen(img_small, caption=f"🔽 Reescalado (Reducido) - {img_small.shape[1]}x{img_small.shape[0]}", ancho=ANCHO_BASE // 2)

            # TODAS LAS DEMÁS con mismo tamaño que original
            img_eq = equalize_image(img)
            mostrar_imagen(img_eq, caption="🌈 Ecualización", ancho=ANCHO_BASE)

            img_linear = apply_linear_filter(img)
            mostrar_imagen(img_linear, caption="📏 Filtro Lineal", ancho=ANCHO_BASE)

            img_nonlinear = apply_nonlinear_filter(img)
            mostrar_imagen(img_nonlinear, caption="📐 Filtro No Lineal", ancho=ANCHO_BASE)

            img_translated = translate_image(img)
            mostrar_imagen(img_translated, caption="↗️ Traslación", ancho=ANCHO_BASE)

            img_rotated = rotate_image(img)
            mostrar_imagen(img_rotated, caption="🔄 Rotación", ancho=ANCHO_BASE)

