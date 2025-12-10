import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

try:
    model = tf.keras.models.load_model(
        "model_Final.keras",
        custom_objects={
            "preprocess_input": tf.keras.applications.resnet50.preprocess_input
        }
    )
    with open('labels.json', 'r') as f:
        class_label = json.load(f)
except Exception as e:
    print(f"Error loading model/labels: {e}")
    class_label = ["plastic", "paper", "metal", "organic", "trash"] 

deskripsi_sampah = {
  "E-waste": {
    "deskripsi": "Sampah elektronik berbahaya. Jangan buang di tempat sampah biasa. Bawa ke drop box e-waste.",
    "tong_warna": "Merah (B3)",
    "dapat_didaur_ulang": "Ya, melalui fasilitas khusus B3",
    "dampak_jika_tidak_diolah": "Pelepasan zat beracun (merkuri, timbal, kadmium) ke tanah dan air. Zat ini mencemari rantai makanan dan sangat berbahaya bagi kesehatan manusia. (Referensi: UNEP/Basel Convention)"
  },
  "Glass": {
    "deskripsi": "Kaca bisa didaur ulang tanpa batas. Pastikan tidak pecah saat dibuang agar aman bagi petugas.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Tidak terurai (inert) dan memenuhi TPA. Pecahan kaca dapat melukai hewan dan petugas, serta berpotensi menyebabkan kebakaran karena efek lensa. (Referensi: Ilmu Lingkungan Material)"
  },
  "Organic Waste": {
    "deskripsi": "Sampah organik (sisa makanan/daun). Bagus untuk dijadikan kompos.",
    "tong_warna": "Hijau (Organik)",
    "dapat_didaur_ulang": "Ya (Diolah menjadi kompos)",
    "dampak_jika_tidak_diolah": "Dalam TPA, penguraian anaerobik menghasilkan gas **metana ($CH_4$)**, yaitu gas rumah kaca yang 25 kali lebih kuat dari karbon dioksida ($CO_2$) dalam memerangkap panas. (Referensi: IPCC/Lembaga Penelitian Lingkungan)"
  },
  "Textiles": {
    "deskripsi": "Limbah tekstil seperti baju bekas. Bisa disumbangkan atau didaur ulang menjadi kain lap.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya (Didaur ulang/Digunakan kembali)",
    "dampak_jika_tidak_diolah": "Membutuhkan lahan TPA yang besar. Tekstil modern melepaskan **serat mikroplastik** saat terurai di lingkungan dan membutuhkan waktu puluhan hingga ratusan tahun. (Referensi: Studi Limbah Tekstil/Microplastic Research)"
  },
  "cardboard": {
    "deskripsi": "Kardus/Karton. Lipat hingga pipih sebelum dibuang untuk menghemat ruang. Bisa didaur ulang menjadi kertas.",
    "tong_warna": "Biru (Kertas)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Memenuhi TPA dan penguraiannya di TPA juga dapat menghasilkan metana jika basah. Daur ulang kardus menghemat energi dan mengurangi penebangan pohon. (Referensi: WWF/Pusat Daur Ulang Kertas)"
  },
  "metal": {
    "deskripsi": "Logam/Kaleng. Cuci bersih sisa makanan sebelum dibuang ke tempat daur ulang.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Logam membutuhkan waktu ratusan tahun untuk terurai. Logam yang berkarat dapat mencemari air tanah dan memerlukan ekstraksi sumber daya alam (penambangan) yang intensif energi. (Referensi: US Geological Survey/Ilmu Material)"
  },
  "paper": {
    "deskripsi": "Kertas. Pastikan kering dan tidak berminyak agar bisa didaur ulang.",
    "tong_warna": "Biru (Kertas)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Menyumbang volume besar di TPA. Kegagalan mendaur ulang berarti peningkatan permintaan kayu dan energi untuk memproduksi kertas baru. (Referensi: Studi Konservasi Energi dan Sumber Daya Alam)"
  },
  "plastic": {
    "deskripsi": "Plastik butuh waktu lama terurai. Pisahkan botol dan gelas plastik untuk didaur ulang.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya",
    "dampak_jika_tidak_diolah": "Membutuhkan ratusan hingga ribuan tahun untuk terurai, mencemari lautan, dan terpecah menjadi **mikroplastik** yang masuk ke rantai makanan dan ekosistem. (Referensi: Jurnal Ilmu Kelautan/Plastics Pollution Coalition)"
  },
  "shoes": {
    "deskripsi": "Sepatu bekas. Jika masih layak pakai, sebaiknya didonasikan.",
    "tong_warna": "Kuning (Anorganik)",
    "dapat_didaur_ulang": "Ya (Digunakan kembali/Daur ulang terbatas)",
    "dampak_jika_tidak_diolah": "Terbuat dari material campuran kompleks (karet, kulit, plastik, busa) yang hampir mustahil terurai secara alami, sehingga menumpuk di TPA. (Referensi: Analisis Material Limbah Kompleks)"
  },
  "trash": {
    "deskripsi": "Sampah residu atau lainnya yang sulit didaur ulang. Buang ke tempat sampah umum.",
    "tong_warna": "Abu-abu (Residu)",
    "dapat_didaur_ulang": "Tidak",
    "dampak_jika_tidak_diolah": "Menyebabkan penumpukan di TPA, memerlukan lahan yang terus bertambah, dan menjadi sumber bau tidak sedap, serta lindi (air sampah) yang mencemari lingkungan. (Referensi: Pedoman Pengelolaan TPA)"
  }
}

def predict_input(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img) 

    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    hasil_label = class_label[idx]
    
    info = deskripsi_sampah.get(hasil_label)
    
    if info:
        deskripsi_markdown = f"""
### Hasil Deteksi: **{hasil_label}**

* **📄 Deskripsi:** {info['deskripsi']}
* **🗑️ Buang Pada Tong Sampah:** {info['tong_warna']}
* **♻️ Dapat Didaur Ulang:** {info['dapat_didaur_ulang']}

---
#### ⚠️ Dampak Jika Tidak Diolah:
{info['dampak_jika_tidak_diolah']}
"""
    else:
        deskripsi_markdown = f"### {hasil_label}\ninformasi detail untuk kategori ini belum tersedia."

    return {class_label[i]: float(pred[i]) for i in range(len(class_label))}, deskripsi_markdown



demo = gr.Interface(
    fn=predict_input,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Prediksi Kategori"), 
        gr.Markdown(label="Saran Pengolahan") 
    ],
    title="Klasifikasi Sampah & Saran Pengolahan",
    description="Unggah foto sampah untuk mengetahui jenisnya dan cara mengolahnya",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    demo.launch(server_name="0.0.0.0", server_port=port)