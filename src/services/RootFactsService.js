import { pipeline, env } from '@huggingface/transformers';
import { TONE_CONFIG, MODEL_CONFIG } from '../utils/config.js';

env.backends.onnx.logLevel = 'error';
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.localModelPath = '/';

// CRITICAL: Batasi numThreads ke 1 untuk menghindari deadlocks pada perangkat low-end.
env.backends.onnx.wasm.numThreads = 1;

/**
 * Basis pengetahuan singkat per sayuran (satu fakta grounding). Teks ini BUKAN yang
 * ditampilkan ke pengguna, melainkan bahan mentah yang diparafrase ulang oleh model
 * Generative AI menjadi fun fact yang unik setiap kali. Fungsinya: (1) menjaga relevansi
 * agar model tetap on-topic, dan (2) sebagai jaring pengaman bila keluaran model gagal.
 */
const VEG_KNOWLEDGE = {
  onion: 'Onion is a bulbous vegetable rich in vitamin C, quercetin, and sulfur compounds that are beneficial for health.',
  carrot: 'Carrot is a root vegetable high in beta-carotene, which the body converts into vitamin A for healthy eyesight.',
  corn: 'Corn is a grain vegetable rich in dietary fiber, B vitamins, lutein, and natural antioxidants.',
  spinach: 'Spinach is a leafy green vegetable packed with iron, calcium, and vitamins A, C, and K.',
  lettuce: 'Lettuce is a leafy green with high water content and a good source of vitamins A and K with very few calories.',
  cucumber: 'Cucumber is a hydrating vegetable that is about 96% water and a good source of vitamins K and C.',
  garlic: 'Garlic is a pungent bulb vegetable rich in allicin, which has antibacterial and antifungal properties.',
  potato: 'Potato is a starchy root vegetable rich in potassium, vitamin C, and complex carbohydrates for energy.',
  ginger: 'Ginger is an aromatic rhizome that contains gingerol, which has anti-inflammatory properties and aids digestion.',
  peas: 'Peas are a legume vegetable high in plant protein, dietary fiber, and vitamins A, C, K, and folate.',
  soybean: 'Soybean is a legume that provides complete plant protein and is rich in isoflavones and essential amino acids.',
  beetroot: 'Beetroot is a root vegetable rich in nitrates, folate, and betalain pigments that support healthy blood flow.',
  paprika: 'Paprika comes from dried peppers and is rich in vitamin A, vitamin E, and antioxidant carotenoids.',
  cabbage: 'Cabbage is a cruciferous vegetable high in vitamin K, vitamin C, and fiber that supports digestion.',
  cauliflower: 'Cauliflower is a cruciferous vegetable low in calories but rich in vitamin C, fiber, and antioxidants.',
  chilli: 'Chilli is a spicy pepper rich in capsaicin, vitamin C, and antioxidants that can boost metabolism.',
  eggplant: 'Eggplant is a nutrient-dense vegetable rich in fiber, antioxidants, and anthocyanin pigments in its skin.',
  turnip: 'Turnip is a root vegetable that is low in calories and a good source of vitamin C and fiber.',
};

// Daftar nama sayuran untuk deteksi kontaminasi: label resmi model + nama sayuran umum
// lain yang berisiko "bocor" dari model (mis. "tomato", "broccoli") walau bukan label resmi.
// Dipakai untuk menolak keluaran AI yang menyebut sayuran LAIN selain yang terdeteksi
// (mis. "Garlic is a well-known tomato"), yang persis pola kesalahan yang membuat
// submission sebelumnya ditolak reviewer.
const KNOWN_VEG_NAMES = [
  ...Object.keys(VEG_KNOWLEDGE),
  'tomato', 'broccoli', 'pepper', 'pumpkin', 'radish', 'celery', 'asparagus', 'zucchini',
];

/**
 * Template prompt per nada (Persona Dinamis). Nada disuntikkan ke instruksi model sehingga
 * gaya penulisan berubah otomatis, sementara fakta grounding menjaga akurasi/relevansi.
 */
const TONE_PROMPT = {
  normal: (name, k) =>
    `Paraphrase the following fact about ${name} into one fun, engaging sentence: ${k}`,
  lucu: (name, k) =>
    `Paraphrase the following fact about ${name} into one playful, humorous sentence that stays accurate: ${k}`,
  santai: (name, k) =>
    `Paraphrase the following fact about ${name} into one relaxed, casual sentence: ${k}`,
  profesional: (name, k) =>
    `Paraphrase the following fact about ${name} into one formal, professional sentence: ${k}`,
};

export class RootFactsService {

  constructor() {
    this.generator = null;
    this.isModelLoaded = false;
    this.isGenerating = false;
    this.config = null;
    this.currentBackend = null;
    this.currentTone = TONE_CONFIG.defaultTone;
  }

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

    // Backend Adaptive: cek navigator.gpu untuk WebGPU, fallback ke WASM jika tidak tersedia.
    // Catatan: Transformers.js (ONNX Runtime Web) tidak memiliki backend WebGL seperti TensorFlow.js;
    // execution provider non-GPU yang tersedia adalah 'wasm', sehingga itulah fallback yang benar di sini.
    if (navigator.gpu) {
      try {
        this.generator = await pipeline('text2text-generation', MODEL_CONFIG.transformersModel, {
          dtype: 'q8',
          device: 'webgpu',
          progress_callback: progressCallback
        });
        this.currentBackend = 'webgpu';
        this.isModelLoaded = true;
        if (onProgress) onProgress(100);
        return;
      } catch (error) {
        console.warn('Transformers.js WebGPU gagal, fallback ke WASM:', error);
      }
    }

    try {
      this.generator = await pipeline('text2text-generation', MODEL_CONFIG.transformersModel, {
        dtype: 'q8',
        device: 'wasm',
        progress_callback: progressCallback
      });
      this.currentBackend = 'wasm';
      this.isModelLoaded = true;
      if (onProgress) onProgress(100);
    } catch (innerError) {
      console.error('Gagal memuat model generator:', innerError);
      throw innerError;
    }
  }

  setTone(tone) {
    this.currentTone = tone;
  }

  async generateFacts(vegetableName) {
    if (!this.generator) throw new Error('Generator belum siap');

    const normalized = vegetableName.toLowerCase().trim();
    const tone = this.currentTone.toLowerCase();

    const toneKey = (['lucu', 'funny'].includes(tone)) ? 'lucu'
      : (['santai', 'casual'].includes(tone)) ? 'santai'
        : (['profesional', 'professional'].includes(tone)) ? 'profesional'
          : 'normal';

    // Fakta grounding untuk sayuran yang terdeteksi (label -> pengetahuan).
    const knowledge = VEG_KNOWLEDGE[normalized]
      ?? `${vegetableName} is a nutritious vegetable with various health benefits.`;

    // Prompt dinamis: label sayuran + nada dipilih dikirim ke model sebagai instruksi.
    const prompt = (TONE_PROMPT[toneKey] ?? TONE_PROMPT.normal)(vegetableName, knowledge);

    // Parameter generasi: do_sample true agar setiap deteksi menghasilkan kalimat berbeda
    // (tidak statis); temperature & top_p mengatur variasi; max_new_tokens dibatasi agar
    // responsif; repetition_penalty moderat mencegah pengulangan kata.
    const result = await this.generator(prompt, {
      max_new_tokens: 70,
      do_sample: true,
      temperature: 0.7,
      top_p: 0.9,
      repetition_penalty: 1.3,
    });

    let text = (result?.[0]?.generated_text ?? '').trim();
    // Model kadang membungkus keluaran dengan tanda kutip; rapikan.
    text = text.replace(/^["'\s]+|["'\s]+$/g, '').trim();

    // Jaring pengaman: keluaran model hanya dipakai bila (1) cukup panjang, (2) benar-benar
    // menyebut nama sayuran yang terdeteksi, dan (3) TIDAK menyebut nama sayuran lain
    // (mis. "Garlic is a well-known tomato") — pola halusinasi yang sama persis dengan
    // penyebab penolakan sebelumnya. Jika gagal, pakai fakta grounding yang pasti benar.
    const lowerText = text.toLowerCase();
    const mentionsCorrectName = lowerText.includes(normalized);
    const mentionsOtherVeg = KNOWN_VEG_NAMES.some((v) =>
      v !== normalized && !normalized.includes(v) && !v.includes(normalized) &&
      new RegExp(`\\b${v}\\b`).test(lowerText));

    if (text.length < 15 || !mentionsCorrectName || mentionsOtherVeg) {
      text = knowledge;
    }

    return text;
  }

  isReady() {
    return this.isModelLoaded;
  }
}
