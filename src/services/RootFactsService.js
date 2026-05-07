import { pipeline } from '@huggingface/transformers';
import { TONE_CONFIG } from '../utils/config.js';

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
    try {
      this.generator = await pipeline('text2text-generation', 'Xenova/flan-t5-small', {
        dtype: 'q4',
        device: 'webgpu',
        progress_callback: (info) => {
          if (info.status === 'progress' && onProgress) {
            onProgress(Math.floor(info.progress));
          }
        }
      });
      this.isModelLoaded = true;
    } catch (error) {
      console.warn('Transformers.js WebGPU gagal, menggunakan CPU:', error);
      this.generator = await pipeline('text2text-generation', 'Xenova/flan-t5-small', {
        dtype: 'q4'
      });
      this.isModelLoaded = true;
    }
  }

  setTone(tone) {
    this.currentTone = tone;
  }

  /**
   * Generasi fun fact menggunakan Persona Dinamis (kriteria Advanced).
   * Prompt diatur dalam Bahasa Inggris sesuai limitasi model.
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
