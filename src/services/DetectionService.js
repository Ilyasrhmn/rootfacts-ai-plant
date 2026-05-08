import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-webgl';

export class DetectionService {
  constructor() {
    this.model = null;
    this.labels = [];
    this.config = {
      modelUrl: '/model/model.json',
      metadataUrl: '/model/metadata.json'
    };
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

      const metadataResponse = await fetch(this.config.metadataUrl);
      const metadata = await metadataResponse.json();
      this.labels = metadata.labels;

      let lastProgress = 0;
      this.model = await tf.loadLayersModel(this.config.modelUrl, {
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
      console.error('Gagal memuat model deteksi:', error);
      throw error;
    }
  }

  /**
   * Melakukan prediksi dengan manajemen memori tf.tidy()
   */
  async predict(imageElement) {
    if (!this.model) return null;

    return tf.tidy(() => {
      const tensor = tf.browser.fromPixels(imageElement)
        .resizeBilinear([224, 224])
        .div(255.0)
        .expandDims(0);

      const prediction = this.model.predict(tensor);
      const probabilities = prediction.dataSync();
      const maxIdx = probabilities.indexOf(Math.max(...probabilities));

      return {
        className: this.labels[maxIdx],
        score: probabilities[maxIdx]
      };
    });
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

