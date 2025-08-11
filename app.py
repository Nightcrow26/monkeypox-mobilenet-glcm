import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

def extract_glcm_features(image):
    image_array = np.array(image)

    # Gunakan PIL untuk konversi ke grayscale
    image_pil = Image.fromarray(image_array)
    gray_image = image_pil.convert("L")
    gray_image_uint = np.array(gray_image).astype(np.uint8)

    glcm = graycomatrix(gray_image_uint, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()

    return np.concatenate([contrast, energy, correlation, homogeneity])

# Mapping label
class_mapping = {
    0: 'Chickenpox',
    1: 'Cowpox',
    2: 'HFMD',
    3: 'Healthy',
    4: 'Measles',
    5: 'Monkeypox'
}

# Tambahkan di bagian atas (setelah class_mapping)
disease_explanations = {
    "Chickenpox": (
        "Cacar air (varicella) biasanya ditandai ruam yang berubah menjadi "
        "lepuh berisi cairan dan kemudian berkeropeng; sering bermula di dada, punggung, dan wajah."
    ),
    "Cowpox": (
        "Cowpox adalah infeksi zoonosis langka; pada manusia umumnya menimbulkan "
        "lesi kulit lokal, sering di tangan/ wajah, terkait paparan hewan terinfeksi."
    ),
    "HFMD": (
        "Hand, Foot, and Mouth Disease (HFMD) umumnya menyebabkan demam ringan, "
        "sariawan di mulut, dan ruam/lepuh pada telapak tangan dan kaki; kebanyakan pulih 7–10 hari."
    ),
    "Healthy": (
        "Tidak tampak ciri khas penyakit pada citra; namun hasil ini bersifat indikatif dari model."
    ),
    "Measles": (
        "Campak umumnya diawali demam, batuk, pilek, mata merah; diikuti ruam "
        "merah yang mulai dari wajah lalu menyebar ke tubuh."
    ),
    "Monkeypox": (
        "Mpox (Monkeypox) ditandai ruam/lesi yang dapat nyeri, sering disertai "
        "pembesaran kelenjar getah bening, demam, sakit kepala, dan lelah."
    ),
}


# Judul aplikasi
st.title("Prediksi Penyakit Kulit Menggunakan Deep Learning + GLCM")
st.write("Unggah gambar kulit untuk diprediksi jenis penyakitnya.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan file sementara
    with open("temp_image." + uploaded_file.name.split('.')[-1], "wb") as f:
        f.write(uploaded_file.getbuffer())
    image_path = "temp_image." + uploaded_file.name.split('.')[-1]

    # Pra-pemrosesan gambar
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array_input = np.expand_dims(img_array, axis=0)

    # Baca dengan OpenCV untuk GLCM
    image_bgr = cv2.imread(image_path)
    glcm_features = extract_glcm_features(image_bgr)
    glcm_features = np.expand_dims(glcm_features, axis=0)

    # Tampilkan gambar
    st.subheader("Gambar Input")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title('RGB Image')
    ax[0].axis('off')

    gray_image = img.convert('L')
    ax[1].imshow(gray_image, cmap='gray')
    ax[1].set_title('Grayscale Image')
    ax[1].axis('off')

    st.pyplot(fig)

    # Load model
    if st.button("Prediksi"):
        with st.spinner("Memuat model..."):
            try:
                model = load_model("feature_model.h5")
            except Exception as e:
                st.error(f"Gagal memuat model: {e}")
                st.stop()

        with st.spinner("Melakukan prediksi..."):
            predictions = model.predict([img_array_input, glcm_features])
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_label = class_mapping[predicted_class_index]
            confidence = np.max(predictions[0]) * 100
           

            # Tampilkan hasil
            if predicted_class_label == "Monkeypox":
                st.error(f"Prediksi: **{predicted_class_label}**")
            else:
                st.success(f"Prediksi: **{predicted_class_label}**")

             st.info(f"Akurasi Prediksi: **{confidence:.2f}%**")
            
            # Tambahkan penjelasan singkat penyakit
            st.subheader("Ringkasan Diagnosis Model")
            st.write(disease_explanations.get(predicted_class_label, "Ringkasan belum tersedia."))


            # Tampilkan properti GLCM
            st.subheader("Fitur GLCM")
            st.write(f"Contrast: {glcm_features[0][0]:.4f}")
            st.write(f"Energy: {glcm_features[0][1]:.4f}")
            st.write(f"Correlation: {glcm_features[0][2]:.4f}")
            st.write(f"Homogeneity: {glcm_features[0][3]:.4f}")

            # Tampilkan probabilitas tiap kelas
            st.subheader("Probabilitas Tiap Kelas")
            probs = {class_mapping[i]: float(predictions[0][i]) for i in range(len(class_mapping))}
            st.bar_chart(probs)

