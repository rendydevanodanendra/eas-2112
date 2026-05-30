import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Penilai Jawaban Otomatis", page_icon="📝")

# 2. Fungsi Load Model (Di-cache agar cepat dan hemat memori)
@st.cache_resource
def load_model():
    # Load dari folder lokal 'model_aes' yang kamu upload
    tokenizer = AutoTokenizer.from_pretrained("./model_aes")
    model = AutoModelForSequenceClassification.from_pretrained("./model_aes")
    return tokenizer, model

# Inisialisasi tokenizer dan model
with st.spinner("Memuat model IndoBERT... (Ini mungkin memakan waktu sebentar)"):
    tokenizer, model = load_model()

# 3. Fungsi Prediksi (Sama seperti di Notebook)
def prediksi(jawaban, kunci):
    # Karena Streamlit Cloud gratis tidak ada GPU, kita paksa pakai CPU
    device = torch.device("cpu")
    model.to(device)
    
    inputs = tokenizer(
        jawaban.lower(),
        kunci.lower(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        skor = outputs.logits.squeeze().item()
    
    # Membatasi rentang 0 - 1, lalu dikali 5 sesuai format nilaimu
    skor = max(0, min(1, skor))
    return skor * 5

# 4. Antarmuka Streamlit
st.title("📝 Penilai Jawaban Otomatis (IndoBERT)")
st.write("Aplikasi ini membandingkan **Jawaban Siswa** dengan **Kunci Jawaban** menggunakan AI untuk memberikan skor 0 hingga 5.")

st.markdown("---")

kunci_jawaban = st.text_area("Kunci Jawaban (Referensi):", height=150, placeholder="Masukkan kunci jawaban ideal di sini...")
jawaban_siswa = st.text_area("Jawaban Siswa:", height=150, placeholder="Masukkan jawaban dari siswa di sini...")

if st.button("Hitung Nilai", type="primary"):
    if kunci_jawaban.strip() == "" or jawaban_siswa.strip() == "":
        st.warning("Mohon isi kedua kolom teks di atas sebelum menilai.")
    else:
        skor_akhir = prediksi(jawaban_siswa, kunci_jawaban)
        
        st.success("Berhasil dihitung!")
        st.metric(label="Nilai Prediksi", value=f"{skor_akhir:.2f} / 5.00")
        
        # Tambahan visual feedback ringan
        if skor_akhir >= 4.0:
            st.info("💡 Jawaban sangat mendekati kunci jawaban.")
        elif skor_akhir >= 2.5:
            st.info("💡 Jawaban cukup baik, tapi masih ada konsep yang terlewat.")
        else:
            st.info("💡 Jawaban kurang tepat atau melenceng dari referensi.")