import { pipeline, env } from '@huggingface/transformers';
import { TONE_CONFIG, MODEL_CONFIG } from '../utils/config.js';

// Konfigurasi Environment Transformers.js
env.backends.onnx.logLevel = 'error';
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.localModelPath = '/';

/**
 * CRITICAL: Batasi numThreads ke 1 untuk menghindari deadlocks 
 * dan 'NetworkError' saat interupsi Service Worker pada perangkat low-end.
 */
env.backends.onnx.wasm.numThreads = 1;

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

    // Mapping Tone dari UI ke deskriptor
    const toneMap = {
      'normal': 'educational and factual',
      'funny': 'funny and witty',
      'lucu': 'funny and witty',
      'casual': 'relaxed and informal',
      'santai': 'relaxed and informal',
      'professional': 'formal and scientific',
      'profesional': 'formal and scientific'
    };

    const safeTone = toneMap[this.currentTone.toLowerCase()] || 'educational and factual';

    const prompt = `Question: Tell me one short, ${safeTone} fact about the plant called ${vegetableName}. \nAnswer:`;

    const result = await this.generator(prompt, {
      max_new_tokens: 40,
      temperature: 0.1,
      do_sample: true,
      top_p: 0.9,
      repetition_penalty: 2.0
    });

    return result[0].generated_text.trim();
  }

  isReady() {
    return this.isModelLoaded;
  }
}
