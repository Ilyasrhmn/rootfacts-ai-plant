import { pipeline, env } from '@huggingface/transformers';
import { TONE_CONFIG, MODEL_CONFIG } from '../utils/config.js';

// Konfigurasi Environment Transformers.js
env.backends.onnx.logLevel = 'error';
env.allowLocalModels = true;
env.allowRemoteModels = true;
env.localModelPath = '/'; 

export class RootFactsService {
  constructor() {
    this.generator = null;
    this.isModelLoaded = false;
    this.currentTone = TONE_CONFIG.defaultTone;
  }

  /**
   * Inisialisasi Transformers.js dengan kuantisasi q4 untuk efisiensi memori.
   */
  async loadModel(onProgress) {
    const progressMap = new Map();
    let lastTotalProgress = 0;

    const progressCallback = (info) => {
      if (info.status === 'progress') {
        progressMap.set(info.file, info.progress);
        
        const totalProgress = Array.from(progressMap.values()).reduce((a, b) => a + b, 0) / progressMap.size;
        const roundedProgress = Math.floor(totalProgress);
        
        if (onProgress && roundedProgress > lastTotalProgress) {
          lastTotalProgress = roundedProgress;
          onProgress(roundedProgress);
        }
      }
    };

    try {
      this.generator = await pipeline('text2text-generation', MODEL_CONFIG.transformersModel, {
        dtype: 'q4',
        device: 'webgpu',
        progress_callback: progressCallback
      });
      this.isModelLoaded = true;
      if (onProgress) onProgress(100);
    } catch (error) {
      console.warn('Transformers.js WebGPU gagal, menggunakan CPU:', error);
      try {
        this.generator = await pipeline('text2text-generation', MODEL_CONFIG.transformersModel, {
          dtype: 'q4',
          progress_callback: progressCallback
        });
        this.isModelLoaded = true;
        if (onProgress) onProgress(100);
      } catch (innerError) {
        console.error('Gagal memuat model generator:', innerError);
        throw innerError;
      }
    }
  }

  setTone(tone) {
    this.currentTone = tone;
  }

  /**
   * Generasi fun fact menggunakan Persona Dinamis (kriteria Advanced).
   */
  async generateFacts(vegetableName) {
    if (!this.generator) throw new Error('Generator belum siap');

    const prompt = `Context: You are a ${this.currentTone} expert on nutrition and plants. 
    Task: Tell me one unique fun fact about ${vegetableName} in a ${this.currentTone} way. 
    Language: English. 
    Fact:`;

    const result = await this.generator(prompt, {
      max_new_tokens: 150,
      temperature: 0.7,
      do_sample: true,
      top_p: 0.9,
    });

    return result[0].generated_text;
  }

  isReady() {
    return this.isModelLoaded;
  }
}
