import { pipeline, env } from '@huggingface/transformers';
import { TONE_CONFIG, MODEL_CONFIG } from '../utils/config.js';

env.backends.onnx.logLevel = 'error';
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.localModelPath = '/';

// CRITICAL: Batasi numThreads ke 1 untuk menghindari deadlocks pada perangkat low-end.
env.backends.onnx.wasm.numThreads = 1;

export class RootFactsService {

  constructor() {
    this.generator = null;
    this.isModelLoaded = false;
    this.isGenerating = false;
    this.config = null;
    this.currentBackend = null;
    this.currentTone = TONE_CONFIG.defaultTone;
  }

  // TODO [Basic] Muat model dan inisialisasi pipeline text2text-generation
  // TODO [Advance] Implementasikan strategi Backend Adaptive
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

    // Backend Adaptive: coba WebGPU dulu, fallback ke CPU jika tidak tersedia.
    try {
      this.generator = await pipeline('text2text-generation', MODEL_CONFIG.transformersModel, {
        dtype: 'q4',
        device: 'webgpu',
        progress_callback: progressCallback
      });
      this.currentBackend = 'webgpu';
      this.isModelLoaded = true;
      if (onProgress) onProgress(100);
    } catch (error) {
      console.warn('Transformers.js WebGPU gagal, menggunakan CPU:', error);
      try {
        this.generator = await pipeline('text2text-generation', MODEL_CONFIG.transformersModel, {
          dtype: 'q4',
          progress_callback: progressCallback
        });
        this.currentBackend = 'cpu';
        this.isModelLoaded = true;
        if (onProgress) onProgress(100);
      } catch (innerError) {
        console.error('Gagal memuat model generator:', innerError);
        throw innerError;
      }
    }
  }

  // TODO [Advance] Konfigurasi tone fakta yang dihasilkan
  setTone(tone) {
    this.currentTone = tone;
  }

  // TODO [Basic] Lakukan prediksi pada elemen gambar yang diberikan dan kembalikan hasilnya
  // TODO [Skilled] Konfigurasikan parameter generasi berdasarkan kebutuhan
  // TODO [Advance] Implemenasikan parameter tone untuk mengatur nada fakta yang dihasilkan
  async generateFacts(vegetableName) {
    if (!this.generator) throw new Error('Generator belum siap');

    // Tone-aware context: nada disematkan dalam teks konteks agar AI mewarisi gayanya saat summarisasi.
    const toneContextMap = {
      'onion': {
        'normal':      'Onion is Allium cepa, a bulbous vegetable rich in vitamin C, quercetin, and sulfur compounds that are beneficial for health.',
        'lucu':        'Onion is that rude vegetable that makes everyone cry in the kitchen, yet we keep inviting it to every meal!',
        'santai':      'Onion is basically a layered veggie that is packed with good stuff and great in just about any dish.',
        'profesional': 'Allium cepa is a monocot bulbous plant rich in organosulfur compounds, flavonoids, and polyphenolic antioxidants.',
      },
      'carrot': {
        'normal':      'Carrot is a root vegetable high in beta-carotene, which the body converts into vitamin A for healthy eyesight.',
        'lucu':        'Carrot is the vegetable that rabbits are obsessed with, and honestly, who can blame them?',
        'santai':      'Carrot is basically a sweet orange root that is super healthy and delicious whether raw or cooked.',
        'profesional': 'Daucus carota subsp. sativus contains elevated concentrations of alpha-carotene, lutein, and polyacetylene compounds.',
      },
      'corn': {
        'normal':      'Corn is a grain vegetable rich in dietary fiber, B vitamins, lutein, and natural antioxidants for good health.',
        'lucu':        'Corn is that sneaky vegetable that always gets stuck in your teeth after eating, no matter how careful you are!',
        'santai':      'Corn is basically the golden vegetable everyone loves because it is sweet, crunchy, and super versatile.',
        'profesional': 'Zea mays is a monocot cereal crop with high starch content, essential amino acids, and carotenoid pigments.',
      },
      'tomato': {
        'normal':      'Tomato is rich in lycopene, potassium, and vitamin C, and is technically a fruit often used as a vegetable.',
        'lucu':        'Tomato is the ultimate identity crisis food—botanists say fruit, chefs say vegetable, and everyone argues at dinner!',
        'santai':      'Tomato is basically the red superstar of the kitchen that is juicy, versatile, and always delicious.',
        'profesional': 'Solanum lycopersicum is a berry-type fruit containing significant carotenoid pigments and phenolic compounds.',
      },
      'spinach': {
        'normal':      'Spinach is a leafy green vegetable packed with iron, calcium, and vitamins A, C, and K.',
        'lucu':        'Spinach is famous for making Popeye strong, and now every parent uses it to trick their kids into eating healthy!',
        'santai':      'Spinach is basically nature\'s own health supplement that is dark, leafy, and loaded with good nutrients.',
        'profesional': 'Spinacia oleracea contains high concentrations of iron, folate, oxalate, and lipoic acid compounds.',
      },
      'broccoli': {
        'normal':      'Broccoli is a cruciferous vegetable rich in sulforaphane, fiber, and vitamins C and K.',
        'lucu':        'Broccoli looks exactly like tiny green trees, which is why kids prefer to play with it rather than eat it!',
        'santai':      'Broccoli is basically little green florets of pure nutrition that are crispy, tasty, and super versatile.',
        'profesional': 'Brassica oleracea var. italica contains glucosinolates, isothiocyanates, and phenolic antioxidant compounds.',
      },
      'bell pepper': {
        'normal':      'Bell pepper is a colorful vegetable high in vitamin C, antioxidants, and vitamin A for immune health.',
        'lucu':        'Bell pepper comes in red, yellow, and green, making it the most fashionable and colorful vegetable in the kitchen!',
        'santai':      'Bell pepper is basically a sweet and crunchy veggie that is perfect for snacking or adding color to any dish.',
        'profesional': 'Capsicum annuum contains carotenoid pigments, ascorbic acid, and capsaicinoid compounds in varying concentrations.',
      },
      'lettuce': {
        'normal':      'Lettuce is a leafy green with high water content and is a good source of vitamins A and K with very few calories.',
        'lucu':        'Lettuce is basically just fancy crunchy water, yet it somehow makes every salad feel more complete and healthy!',
        'santai':      'Lettuce is basically the light and fresh green that makes salads crispy, cool, and super refreshing.',
        'profesional': 'Lactuca sativa contains high moisture content alongside vitamins K and A, folate, and chlorophyll compounds.',
      },
      'cucumber': {
        'normal':      'Cucumber is a hydrating vegetable that is 96% water and a good source of vitamins K and C.',
        'lucu':        'Cucumber is literally 96% water, which means you are basically eating a very crunchy glass of water!',
        'santai':      'Cucumber is basically a cool and refreshing green that is perfect for salads, snacking, or staying hydrated.',
        'profesional': 'Cucumis sativus is rich in water content, vitamin K, cucurbitacin compounds, and phenolic antioxidants.',
      },
      'garlic': {
        'normal':      'Garlic is a pungent bulb vegetable rich in allicin, which has strong antibacterial and antifungal properties.',
        'lucu':        'Garlic is the vegetable that makes food taste amazing but also keeps friends at a respectful distance afterward!',
        'santai':      'Garlic is basically the secret weapon of cooking that makes everything taste better with just a clove or two.',
        'profesional': 'Allium sativum contains organosulfur compounds including allicin, which exhibit antimicrobial and cardioprotective effects.',
      },
      'potato': {
        'normal':      'Potato is a starchy root vegetable rich in potassium, vitamin C, and complex carbohydrates for energy.',
        'lucu':        'Potato is so versatile that humans have invented over a thousand ways to cook it—we are truly obsessed with it!',
        'santai':      'Potato is basically the ultimate comfort food that can be boiled, baked, mashed, or fried into something amazing.',
        'profesional': 'Solanum tuberosum tubers contain high starch concentrations, potassium, vitamin C, and glycoalkaloid compounds.',
      },
      'ginger': {
        'normal':      'Ginger is an aromatic rhizome that contains gingerol, which has anti-inflammatory properties and aids digestion.',
        'lucu':        'Ginger is that spicy little root that kicks your tongue and then somehow makes your stomach feel totally fine!',
        'santai':      'Ginger is basically the spicy superhero of the kitchen that warms you up and settles your stomach at the same time.',
        'profesional': 'Zingiber officinale rhizomes contain gingerols, shogaols, and paradols with documented anti-inflammatory bioactivity.',
      },
      'peas': {
        'normal':      'Peas are a legume vegetable high in plant protein, dietary fiber, and vitamins A, C, K, and folate.',
        'lucu':        'Peas are tiny green balls of nutrition that kids love to roll around the plate before finally eating them!',
        'santai':      'Peas are basically tiny green powerhouses of protein and vitamins that are sweet, tender, and easy to enjoy.',
        'profesional': 'Pisum sativum seeds contain high concentrations of plant protein, dietary fiber, and folate with low glycemic index.',
      },
      'soybean': {
        'normal':      'Soybean is a legume that provides complete plant protein and is rich in isoflavones and essential amino acids.',
        'lucu':        'Soybean is so versatile that it can become tofu, milk, sauce, or oil—basically the shapeshifter of the food world!',
        'santai':      'Soybean is basically the plant world\'s best source of complete protein that can turn into almost any food product.',
        'profesional': 'Glycine max seeds contain complete protein with all essential amino acids, isoflavones, and polyunsaturated fatty acids.',
      },
    };

    const normalized = vegetableName.toLowerCase().trim();
    const tone = this.currentTone.toLowerCase();

    const toneKey = (['lucu', 'funny'].includes(tone))            ? 'lucu'
                  : (['santai', 'casual'].includes(tone))          ? 'santai'
                  : (['profesional', 'professional'].includes(tone)) ? 'profesional'
                  : 'normal';

    const toneContext = toneContextMap[normalized]?.[toneKey]
      ?? `${vegetableName} is a nutritious vegetable with various health benefits.`;

    const prompt = `summarize: ${toneContext}`;

    const result = await this.generator(prompt, {
      max_new_tokens: 60,
      do_sample: false,
      repetition_penalty: 1.8,
    });

    let text = result[0].generated_text.trim();

    if (!text || text.length < 10) {
      text = toneContext;
    }

    return text;
  }

  // TODO [Basic] Periksa apakah model sudah dimuat dan siap digunakan
  isReady() {
    return this.isModelLoaded;
  }
}