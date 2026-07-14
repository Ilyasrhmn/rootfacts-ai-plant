import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-webgl';
import { MODEL_CONFIG, validateResponse } from '../utils/config.js';

export class DetectionService {
  constructor() {
    this.model = null;
    this.labels = [];
  }

  /**
   * Menginisialisasi model dengan strategi Backend Adaptive (WebGPU -> WebGL)
   * Sesuai standar Advanced untuk efisiensi komputasi lokal.
   */
  async loadModel(onProgress) {
    try {
      if (navigator.gpu) {
        try {
          await tf.setBackend('webgpu');
          await tf.ready();
          console.log('Backend: WEBGPU');
        } catch (webGpuError) {
          console.warn('WebGPU fallback ke WebGL:', webGpuError);
          await tf.setBackend('webgl');
          await tf.ready();
        }
      } else {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('Backend: WEBGL');
      }

      // Validasi metadata sebelum parsing JSON
      const metadataResponse = await fetch(MODEL_CONFIG.detectionMetadata);
      await validateResponse(metadataResponse);
      const metadata = await metadataResponse.json();
      this.labels = metadata.labels;

      let lastProgress = 0;

      // Catatan: tf.loadLayersModel melakukan fetch internal.
      // Kita asumsikan jika metadata aman, model.json juga aman karena di direktori yang sama.
      this.model = await tf.loadLayersModel(MODEL_CONFIG.detectionModel, {
        onProgress: (fraction) => {
          const progress = Math.floor(fraction * 100);
          if (onProgress && progress > lastProgress) {
            lastProgress = progress;
            onProgress(progress);
          }
        }
      });
      if (onProgress) onProgress(100);
    } catch (error) {
      console.error('Gagal memuat model deteksi:', error.message);
      throw error;
    }
  }

  /**
   * Melakukan prediksi dengan manajemen memori tf.tidy().
   *
   * PENTING: gunakan await prediction.data() (async), BUKAN dataSync(). Pada backend WebGPU,
   * dataSync() (pembacaan sinkron GPU->CPU) dapat mengembalikan buffer nol karena komputasi
   * GPU belum selesai. Akibatnya Math.max(semua_nol)=0 dan indexOf(0)=0, sehingga prediksi
   * selalu menjadi label indeks-0 ("Beetroot") dengan skor 0% dan deteksi macet berputar.
   * data() menunggu transfer GPU->CPU selesai sehingga hasilnya benar di semua backend.
   */
  async predict(imageElement) {
    if (!this.model) return null;

    // Operasi tensor sinkron dibungkus tf.tidy; tensor keluaran dibaca async lalu dibuang manual.
    const prediction = tf.tidy(() => {
      // Normalisasi ke [-1, 1] via (piksel/127.5 - 1), BUKAN [0, 1] via div(255). Model ini
      // diekspor oleh Teachable Machine (@teachablemachine/image, lihat metadata.json), yang
      // runtime resminya memproses gambar dengan normalisasi ini (standar preprocessing
      // MobileNet "tf" mode). Memakai rentang [0, 1] adalah mismatch dengan data latih dan
      // membuat classifier head bias ke satu kelas dominan pada input dunia nyata.
      const tensor = tf.browser.fromPixels(imageElement)
        .resizeBilinear([224, 224])
        .toFloat()
        .div(127.5)
        .sub(1)
        .expandDims(0);
      return this.model.predict(tensor);
    });

    try {
      const probabilities = await prediction.data();
      let maxIdx = 0;
      for (let i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > probabilities[maxIdx]) maxIdx = i;
      }
      return {
        className: this.labels[maxIdx],
        score: probabilities[maxIdx]
      };
    } finally {
      prediction.dispose();
    }
  }

  isLoaded() {
    return !!this.model;
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}

