import os
import flask
from flask import request, render_template, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
import base64
import numpy as np
import logging

# Muat environment variables dari file .env
load_dotenv()

# Konfigurasi API Key untuk Gemini
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except TypeError:
    print("GOOGLE_API_KEY tidak ditemukan. Pastikan file .env sudah benar.")

# Menonaktifkan logging berlebihan
app = flask.Flask(__name__)
CORS(app)


def decode_base64_image(data_uri):
    """Mendekode gambar base64 menjadi format array, hanya untuk gimmick (tanpa analisis)."""
    try:
        encoded = data_uri.split(",")[1]
        _ = base64.b64decode(encoded)  # Tidak dipakai, hanya decoding dasar
        return True  # Return dummy True agar tetap ke form
    except (IndexError, TypeError, base64.binascii.Error):
        return False


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/scan')
def scan_page():
    return render_template("scan.html")


@app.route('/show_form', methods=['POST'])
def show_form():
    """Pindai wajah (gimmick saja), langsung tampilkan form dengan skor emosi default."""
    image_data = request.form.get('image_data')
    if image_data:
        decode_base64_image(image_data)  # Tidak digunakan, hanya gimmick
    emotion_score = 50  # Nilai tetap
    return render_template("form.html", emotion_score=emotion_score)


@app.route('/analyze', methods=['POST'])
def analyze():
    q_maps = {
        'q1': {'10': 'Pekerjaan / tugas', '15': 'Hubungan sosial / keluarga', '20': 'Finansial atau Kesehatan', 'default': 'Tidak disebutkan'},
        'q2': {'5': 'Beberapa hari', '10': 'Lebih dari seminggu', '15': 'Beberapa bulan', '20': 'Sangat lama, bahkan lupa kapan merasa baik', 'default': 'Tidak disebutkan'},
        'q3': {'15': 'Cemas / khawatir berlebihan', '10': 'Marah / mudah tersinggung', '20': 'Lelah atau Mati rasa', 'default': 'Tidak disebutkan'},
        'q4': {'0': 'Nyenyak & cukup', '10': 'Sering bangun / gelisah', '15': 'Sulit tidur', '20': 'Terlalu banyak tidur', 'default': 'Tidak disebutkan'},
        'q5': {'0': 'Curhat ke teman / keluarga', '10': 'Menyibukkan diri', '20': 'Menarik diri dari sekitar', '15': 'Bingung harus bagaimana', 'default': 'Tidak disebutkan'},
        'q6': {'0': 'Ya, lebih dari satu', '5': 'Ya, tapi hanya satu-dua', '15': 'Tidak yakin', '20': 'Tidak ada', 'default': 'Tidak disebutkan'},
        'q7': {'5': 'Waktu istirahat atau Tempat curhat', '10': 'Arahan atau solusi praktis', '15': 'Tidak tahu, tapi ingin merasa lebih baik', 'default': 'Tidak disebutkan'}
    }

    emotion_score = int(request.form.get('emotion_score', 50))  # Tetap 50 dari scan
    answers_values = {f'q{i}': request.form.get(f'q{i}') for i in range(1, 8)}
    answers_text = {
        f'q{i}_text': q_maps[f'q{i}'].get(answers_values[f'q{i}'], q_maps[f'q{i}']['default']) 
        for i in range(1, 8)
    }

    questionnaire_score = sum(int(v) for v in answers_values.values() if v is not None)
    max_questionnaire_score = 135
    normalized_q_score = (questionnaire_score / max_questionnaire_score) * 100
    final_stress_score = int((0.6 * normalized_q_score) + (0.4 * emotion_score))

    if final_stress_score >= 70:
        stress_label = "Stres berat"
    elif final_stress_score >= 45:
        stress_label = "Stres sedang"
    else:
        stress_label = "Stres ringan"

    return render_template(
        "result.html",
        percentage=final_stress_score,
        label=stress_label,
        **answers_text
    )


@app.route('/get_solution', methods=['POST'])
def get_solution():
    if not os.environ.get("GOOGLE_API_KEY"):
        return jsonify({'solution': 'Konfigurasi GOOGLE_API_KEY tidak ditemukan.'}), 500

    data = request.json
    score = data.get('score')
    label = data.get('label')
    q1_text = data.get('q1')
    q2_text = data.get('q2')
    q3_text = data.get('q3')
    q4_text = data.get('q4')
    q5_text = data.get('q5')
    q6_text = data.get('q6')
    q7_text = data.get('q7')

    ai_solution = "Gagal mendapatkan saran dari AI. Coba lagi nanti."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        Anda adalah seorang konselor psikologi AI yang empatik, bijak, dan suportif dalam bahasa Indonesia.
        Seorang pengguna baru saja menyelesaikan kuesioner tentang kondisi mental mereka.

        Berikut adalah ringkasan hasilnya:
        - Tingkat Stres Terdeteksi: {label} ({score}%).
        - Pikiran yang paling dominan: "{q1_text}".
        - Durasi perasaan ini: "{q2_text}".
        - Emosi yang paling sering dirasakan: "{q3_text}".
        - Kualitas tidur: "{q4_text}".
        - Cara mengatasi tekanan saat ini: "{q5_text}".
        - Dukungan sosial yang dirasakan: "{q6_text}".
        - Kebutuhan yang paling mendesak: "{q7_text}".

        Tugas Anda:
        Berikan tanggapan yang hangat dan sangat personal berdasarkan jawaban spesifik di atas.
        1. Mulai dengan validasi perasaan mereka, sebutkan 1-2 poin spesifik dari jawaban mereka (Contoh: "Saya mengerti, merasa lelah meski sudah istirahat sambil memikirkan soal finansial pasti sangat menguras energi...").
        2. Berikan 2-3 langkah konkret dan praktis yang relevan dengan masalah yang mereka sebutkan.
        3. Akhiri dengan kalimat penyemangat yang tulus.
        4. Gunakan format poin atau nomor agar mudah dibaca.
        5. PENTING: Jangan sertakan disclaimer tentang bantuan profesional, karena itu sudah ada di UI.
        """
        response = model.generate_content(prompt)
        ai_solution = response.text

    except Exception as e:
        print(f"Error saat memanggil Google AI API: {e}")

    return jsonify({'solution': ai_solution})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
