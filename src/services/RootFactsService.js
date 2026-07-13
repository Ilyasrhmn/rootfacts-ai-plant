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
    // Penghitung rotasi per (sayuran|nada) agar setiap deteksi berturut-turut
    // menghasilkan varian teks yang berbeda (mencegah output terasa statis).
    this.rotationCounters = new Map();
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
          dtype: 'q4',
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
        dtype: 'q4',
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

    // Basis pengetahuan terkurasi. Setiap (sayuran x nada) memiliki BEBERAPA varian,
    // sehingga deteksi berulang tidak menampilkan teks yang sama (tidak statis) dan
    // nama sayuran dijamin benar (menutup celah "Soybean" -> "Soybes"). Nada disematkan
    // di dalam teks agar gaya penulisan berubah sesuai persona yang dipilih pengguna.
    const toneContextMap = {
      'onion': {
        'normal':      ['Onion is Allium cepa, a bulbous vegetable rich in vitamin C, quercetin, and sulfur compounds that are beneficial for health.', 'Onion is a kitchen staple whose layers store natural sugars and antioxidants that help support the immune system.'],
        'lucu':        ['Onion is that rude vegetable that makes everyone cry in the kitchen, yet we keep inviting it to every meal!', 'Onion is the only vegetable brave enough to make you cry and still get invited back for dinner every single night!'],
        'santai':      ['Onion is basically a layered veggie that is packed with good stuff and great in just about any dish.', 'Onion is basically the easygoing veggie that quietly makes every soup, stir-fry, and salad taste better.'],
        'profesional': ['Allium cepa is a monocot bulbous plant rich in organosulfur compounds, flavonoids, and polyphenolic antioxidants.', 'Allium cepa exhibits notable antioxidant capacity owing to its quercetin content and sulfur-based bioactive compounds.'],
      },
      'carrot': {
        'normal':      ['Carrot is a root vegetable high in beta-carotene, which the body converts into vitamin A for healthy eyesight.', 'Carrot is a crunchy root vegetable that supplies fiber, potassium, and antioxidants that support overall health.'],
        'lucu':        ['Carrot is the vegetable that rabbits are obsessed with, and honestly, who can blame them?', 'Carrot is so loved by rabbits that cartoons made it their official snack, which is honestly great branding!'],
        'santai':      ['Carrot is basically a sweet orange root that is super healthy and delicious whether raw or cooked.', 'Carrot is basically the friendly orange snack you can munch raw or toss into almost any cooked dish.'],
        'profesional': ['Daucus carota subsp. sativus contains elevated concentrations of alpha-carotene, lutein, and polyacetylene compounds.', 'Daucus carota is a rich dietary source of provitamin A carotenoids that contribute to visual and immune function.'],
      },
      'corn': {
        'normal':      ['Corn is a grain vegetable rich in dietary fiber, B vitamins, lutein, and natural antioxidants for good health.', 'Corn is a versatile cereal crop that provides energy from complex carbohydrates along with fiber and antioxidants.'],
        'lucu':        ['Corn is that sneaky vegetable that always gets stuck in your teeth after eating, no matter how careful you are!', 'Corn is the only veggie that leaves evidence in your teeth and turns every barbecue into a flossing contest!'],
        'santai':      ['Corn is basically the golden vegetable everyone loves because it is sweet, crunchy, and super versatile.', 'Corn is basically the golden all-rounder that is just as happy grilled, popped, or dropped into a warm soup.'],
        'profesional': ['Zea mays is a monocot cereal crop with high starch content, essential amino acids, and carotenoid pigments.', 'Zea mays provides substantial caloric energy and lutein-zeaxanthin carotenoids associated with ocular health.'],
      },
      'spinach': {
        'normal':      ['Spinach is a leafy green vegetable packed with iron, calcium, and vitamins A, C, and K.', 'Spinach is a nutrient-dense leafy green that delivers folate, iron, and antioxidants with very few calories.'],
        'lucu':        ['Spinach is famous for making Popeye strong, and now every parent uses it to trick their kids into eating healthy!', 'Spinach single-handedly convinced a generation of kids that eating leaves gives you instant muscles, thanks to Popeye!'],
        'santai':      ['Spinach is basically nature\'s own health supplement that is dark, leafy, and loaded with good nutrients.', 'Spinach is basically the chill leafy green you can wilt into pasta or blend into a smoothie without any fuss.'],
        'profesional': ['Spinacia oleracea contains high concentrations of iron, folate, oxalate, and lipoic acid compounds.', 'Spinacia oleracea is a valuable source of non-heme iron, folate, and lutein with recognized antioxidant activity.'],
      },
      'lettuce': {
        'normal':      ['Lettuce is a leafy green with high water content and is a good source of vitamins A and K with very few calories.', 'Lettuce is a crisp, hydrating leafy green that adds vitamins A and K to meals while staying very low in calories.'],
        'lucu':        ['Lettuce is basically just fancy crunchy water, yet it somehow makes every salad feel more complete and healthy!', 'Lettuce is proof that crunchy water can headline an entire salad and still take all the credit for being healthy!'],
        'santai':      ['Lettuce is basically the light and fresh green that makes salads crispy, cool, and super refreshing.', 'Lettuce is basically the easygoing green that keeps sandwiches and salads cool, crunchy, and refreshing.'],
        'profesional': ['Lactuca sativa contains high moisture content alongside vitamins K and A, folate, and chlorophyll compounds.', 'Lactuca sativa is characterized by high water content and modest but useful levels of vitamin K and folate.'],
      },
      'cucumber': {
        'normal':      ['Cucumber is a hydrating vegetable that is 96% water and a good source of vitamins K and C.', 'Cucumber is a refreshing, water-rich vegetable that aids hydration and provides vitamin K and antioxidants.'],
        'lucu':        ['Cucumber is literally 96% water, which means you are basically eating a very crunchy glass of water!', 'Cucumber is basically a snack and a drink at the same time, being 96% water with a satisfying crunch on top!'],
        'santai':      ['Cucumber is basically a cool and refreshing green that is perfect for salads, snacking, or staying hydrated.', 'Cucumber is basically the chill, cooling veggie you reach for on a hot day to snack and stay hydrated.'],
        'profesional': ['Cucumis sativus is rich in water content, vitamin K, cucurbitacin compounds, and phenolic antioxidants.', 'Cucumis sativus offers high hydration value along with vitamin K and cucurbitacin compounds of nutritional interest.'],
      },
      'garlic': {
        'normal':      ['Garlic is a pungent bulb vegetable rich in allicin, which has strong antibacterial and antifungal properties.', 'Garlic is an aromatic bulb whose allicin compound is linked to antibacterial effects and heart health support.'],
        'lucu':        ['Garlic is the vegetable that makes food taste amazing but also keeps friends at a respectful distance afterward!', 'Garlic is the flavor hero that saves your dinner and then quietly ruins your chances of close conversation!'],
        'santai':      ['Garlic is basically the secret weapon of cooking that makes everything taste better with just a clove or two.', 'Garlic is basically the little clove that quietly upgrades any dish from plain to seriously tasty.'],
        'profesional': ['Allium sativum contains organosulfur compounds including allicin, which exhibit antimicrobial and cardioprotective effects.', 'Allium sativum is valued for allicin and related organosulfur compounds with documented cardiovascular benefits.'],
      },
      'potato': {
        'normal':      ['Potato is a starchy root vegetable rich in potassium, vitamin C, and complex carbohydrates for energy.', 'Potato is a filling root vegetable that provides energy from starch along with potassium and vitamin C.'],
        'lucu':        ['Potato is so versatile that humans have invented over a thousand ways to cook it, so we are truly obsessed with it!', 'Potato has more career options than most people, from fries to chips to mash, and honestly it never stops working!'],
        'santai':      ['Potato is basically the ultimate comfort food that can be boiled, baked, mashed, or fried into something amazing.', 'Potato is basically the cozy comfort food that turns into something delicious no matter how you cook it.'],
        'profesional': ['Solanum tuberosum tubers contain high starch concentrations, potassium, vitamin C, and glycoalkaloid compounds.', 'Solanum tuberosum is an energy-dense tuber supplying resistant starch, potassium, and ascorbic acid.'],
      },
      'ginger': {
        'normal':      ['Ginger is an aromatic rhizome that contains gingerol, which has anti-inflammatory properties and aids digestion.', 'Ginger is a spicy rhizome traditionally used to ease nausea and support digestion thanks to its gingerol content.'],
        'lucu':        ['Ginger is that spicy little root that kicks your tongue and then somehow makes your stomach feel totally fine!', 'Ginger is the tiny root that slaps your taste buds awake and then apologizes by settling your upset stomach!'],
        'santai':      ['Ginger is basically the spicy superhero of the kitchen that warms you up and settles your stomach at the same time.', 'Ginger is basically the warm, zingy root that makes teas and stir-fries cozy while calming your tummy.'],
        'profesional': ['Zingiber officinale rhizomes contain gingerols, shogaols, and paradols with documented anti-inflammatory bioactivity.', 'Zingiber officinale is recognized for gingerol and shogaol compounds with anti-inflammatory and antiemetic activity.'],
      },
      'peas': {
        'normal':      ['Peas are a legume vegetable high in plant protein, dietary fiber, and vitamins A, C, K, and folate.', 'Peas are small green legumes that pack plant protein and fiber along with vitamins C and K into every pod.'],
        'lucu':        ['Peas are tiny green balls of nutrition that kids love to roll around the plate before finally eating them!', 'Peas are the tiny green marbles that turn every kid\'s dinner plate into a rolling game before they get eaten!'],
        'santai':      ['Peas are basically tiny green powerhouses of protein and vitamins that are sweet, tender, and easy to enjoy.', 'Peas are basically sweet little green bites that drop easily into rice, pasta, or soup whenever you like.'],
        'profesional': ['Pisum sativum seeds contain high concentrations of plant protein, dietary fiber, and folate with low glycemic index.', 'Pisum sativum provides plant-based protein and soluble fiber with a favorable, relatively low glycemic response.'],
      },
      'soybean': {
        'normal':      ['Soybean is a legume that provides complete plant protein and is rich in isoflavones and essential amino acids.', 'Soybean is a protein-rich legume and one of the few plant foods that supplies all nine essential amino acids.'],
        'lucu':        ['Soybean is so versatile that it can become tofu, milk, sauce, or oil, basically the shapeshifter of the food world!', 'Soybean is the ultimate food-world shapeshifter, one bean that moonlights as tofu, milk, sauce, and even cooking oil!'],
        'santai':      ['Soybean is basically the plant world\'s best source of complete protein that can turn into almost any food product.', 'Soybean is basically the laid-back bean that quietly becomes tofu, milk, or edamame whenever you need it.'],
        'profesional': ['Glycine max seeds contain complete protein with all essential amino acids, isoflavones, and polyunsaturated fatty acids.', 'Glycine max is a complete plant protein source notable for isoflavones and beneficial polyunsaturated fatty acids.'],
      },
      'beetroot': {
        'normal':      ['Beetroot is a root vegetable rich in nitrates, folate, and betalain pigments that support healthy blood flow.', 'Beetroot is a deep-red root vegetable whose natural nitrates are linked to healthy circulation and stamina.'],
        'lucu':        ['Beetroot stains everything it touches bright pink, including your hands, your cutting board, and your dinner plans!', 'Beetroot is the veggie that leaves a bright pink crime scene on your hands, your board, and probably your shirt too!'],
        'santai':      ['Beetroot is basically the earthy purple-red root that is sweet, healthy, and great in salads or juices.', 'Beetroot is basically the mellow, earthy-sweet root that blends smoothly into juices, salads, or roasts.'],
        'profesional': ['Beta vulgaris taproots contain betalain pigments, dietary nitrates, and folate with documented vasodilatory effects.', 'Beta vulgaris is studied for dietary nitrates and betalains associated with vasodilation and antioxidant activity.'],
      },
      'paprika': {
        'normal':      ['Paprika comes from dried peppers and is rich in vitamin A, vitamin E, and antioxidant carotenoids.', 'Paprika is a mild spice ground from dried peppers, offering vitamin A and colorful antioxidant carotenoids.'],
        'lucu':        ['Paprika is basically a pepper that decided to retire from the salad bowl and become a glamorous spice instead!', 'Paprika is what happens when a pepper retires and reinvents itself as the most colorful spice in the rack!'],
        'santai':      ['Paprika is basically the smoky-sweet spice that adds warm color and flavor to almost any dish.', 'Paprika is basically the easygoing spice that sprinkles gentle color and mild flavor over just about anything.'],
        'profesional': ['Capsicum annuum-derived paprika contains capsanthin, carotenoid pigments, and tocopherol antioxidant compounds.', 'Paprika from Capsicum annuum is rich in capsanthin and tocopherols that contribute antioxidant properties.'],
      },
      'cabbage': {
        'normal':      ['Cabbage is a cruciferous vegetable high in vitamin K, vitamin C, and fiber that supports digestion.', 'Cabbage is a leafy cruciferous vegetable that delivers vitamin C, vitamin K, and gut-friendly dietary fiber.'],
        'lucu':        ['Cabbage is basically lettuce that went to the gym and came back wearing tight layered armor!', 'Cabbage is basically lettuce\'s buff cousin, wrapping itself in layer after layer of leafy green armor!'],
        'santai':      ['Cabbage is basically the crunchy leafy veggie that is cheap, healthy, and great in almost any dish.', 'Cabbage is basically the budget-friendly cruncher that shreds happily into slaws, soups, and stir-fries.'],
        'profesional': ['Brassica oleracea var. capitata contains glucosinolates, vitamin K, and soluble fiber compounds.', 'Brassica oleracea var. capitata provides glucosinolates and vitamin K with notable digestive fiber content.'],
      },
      'cauliflower': {
        'normal':      ['Cauliflower is a cruciferous vegetable low in calories but rich in vitamin C, fiber, and antioxidants.', 'Cauliflower is a versatile cruciferous vegetable that is low in calories yet high in vitamin C and fiber.'],
        'lucu':        ['Cauliflower is basically broccoli that bleached itself white and now pretends to be rice or pizza crust!', 'Cauliflower is basically broccoli that went pale and started a second career as rice, pizza crust, and even steak!'],
        'santai':      ['Cauliflower is basically the versatile white veggie that can turn into rice, pizza, or steak these days.', 'Cauliflower is basically the chill white veggie happy to be roasted whole or riced into a light side.'],
        'profesional': ['Brassica oleracea var. botrytis contains glucosinolates, ascorbic acid, and dietary fiber compounds.', 'Brassica oleracea var. botrytis supplies ascorbic acid and glucosinolates with low caloric density.'],
      },
      'chilli': {
        'normal':      ['Chilli is a spicy fruit-vegetable rich in capsaicin, vitamin C, and antioxidants that boost metabolism.', 'Chilli is a fiery pepper whose capsaicin gives it heat and is associated with a temporary metabolism boost.'],
        'lucu':        ['Chilli is the tiny vegetable that can bring grown adults to tears and regret in a single bite!', 'Chilli is the small but mighty pepper that turns confident adults into teary, milk-chugging regret machines!'],
        'santai':      ['Chilli is basically the spicy little kick that makes any dish way more exciting.', 'Chilli is basically the fun little heat boost you add when a dish needs a bit more excitement.'],
        'profesional': ['Capsicum frutescens contains capsaicinoid compounds responsible for pungency and thermogenic bioactivity.', 'Capsicum frutescens owes its pungency to capsaicinoids, which also exhibit thermogenic and antioxidant effects.'],
      },
      'eggplant': {
        'normal':      ['Eggplant is a nutrient-dense vegetable rich in fiber, antioxidants, and anthocyanin pigments in its skin.', 'Eggplant is a fiber-rich vegetable whose glossy purple skin carries antioxidant anthocyanin pigments.'],
        'lucu':        ['Eggplant is the vegetable that looks nothing like an egg, yet somehow kept the name forever!', 'Eggplant is the veggie with the most misleading name ever, with no eggs involved, but the label just stuck!'],
        'santai':      ['Eggplant is basically the purple, spongy veggie that soaks up flavor in almost any dish.', 'Eggplant is basically the mellow purple sponge that happily soaks up sauces and spices while it cooks.'],
        'profesional': ['Solanum melongena contains nasunin anthocyanin pigments, chlorogenic acid, and dietary fiber.', 'Solanum melongena is notable for nasunin and chlorogenic acid, antioxidants concentrated in its skin.'],
      },
      'turnip': {
        'normal':      ['Turnip is a root vegetable that is low in calories and a good source of vitamin C and fiber.', 'Turnip is a humble root vegetable that offers vitamin C and fiber while staying light on calories.'],
        'lucu':        ['Turnip is basically a potato\'s peppery cousin that nobody talks about at family dinners!', 'Turnip is the peppery root that shows up to the family dinner and still nobody remembers its name!'],
        'santai':      ['Turnip is basically the mild peppery root that is great roasted, mashed, or tossed in soups.', 'Turnip is basically the easygoing peppery root that roasts up sweet or slips quietly into a warm stew.'],
        'profesional': ['Brassica rapa taproots contain glucosinolates, ascorbic acid, and moderate dietary fiber content.', 'Brassica rapa taproots provide ascorbic acid and glucosinolates with moderate, digestion-supporting fiber.'],
      },
    };

    const normalized = vegetableName.toLowerCase().trim();
    const tone = this.currentTone.toLowerCase();

    const toneKey = (['lucu', 'funny'].includes(tone))            ? 'lucu'
      : (['santai', 'casual'].includes(tone))          ? 'santai'
        : (['profesional', 'professional'].includes(tone)) ? 'profesional'
          : 'normal';

    const variants = toneContextMap[normalized]?.[toneKey]
      ?? [`${vegetableName} is a nutritious vegetable with various health benefits.`];

    // Rotasi varian per (sayuran|nada): deteksi berturut-turut memilih varian berbeda,
    // sehingga teks tidak pernah terasa statis walau objek yang sama dipindai berkali-kali.
    const rotationKey = `${normalized}|${toneKey}`;
    const counter = this.rotationCounters.get(rotationKey) ?? 0;
    const seedContext = variants[counter % variants.length];
    this.rotationCounters.set(rotationKey, counter + 1);

    // Prompt dinamis: konteks (yang memuat label sayuran terdeteksi + nada) dikirim ke model.
    const prompt = `summarize: ${seedContext}`;

    // Parameter generasi: do_sample true agar keluaran bervariasi antar generasi;
    // temperature & top_p mengatur kadar variasi; repetition_penalty moderat (1.3)
    // dan no_repeat_ngram_size mencegah pengulangan/korupsi kata.
    const result = await this.generator(prompt, {
      max_new_tokens: 80,
      do_sample: true,
      temperature: 0.7,
      top_p: 0.9,
      repetition_penalty: 1.3,
      no_repeat_ngram_size: 3,
    });

    const text = (result?.[0]?.generated_text ?? '').trim();

    // Guard kualitas & kebenaran nama: keluaran model hanya dipakai bila benar-benar layak,
    // yaitu (1) menyebut nama sayuran yang benar, (2) kalimat utuh diakhiri tanda baca,
    // (3) panjang wajar, dan (4) tidak ada kata yang berulang berlebihan (indikasi ngawur).
    // Jika tidak layak (mis. "Soybean" -> "Soybes", teks terpotong, atau pengulangan), pakai
    // fakta terkurasi. Karena seedContext dirotasi, fallback pun tetap bervariasi antar deteksi.
    const lower = text.toLowerCase();
    const words = lower.match(/\b[a-z]{4,}\b/g) ?? [];
    const uniqueWords = [...new Set(words)];

    // (1) menyebut nama sayuran yang benar
    const mentionsName = lower.includes(normalized);
    // (2) kalimat utuh diakhiri tanda baca (menolak teks terpotong seperti "...fatty acids:")
    const endsCleanly = /[.!?]$/.test(text);
    // (3) tidak ada kata yang berulang berlebihan (indikasi keluaran ngawur "this this this")
    const maxRepeat = uniqueWords.reduce(
      (max, w) => Math.max(max, words.filter((x) => x === w).length), 0);
    // (4) tetap "on-topic": berbagi cukup banyak kata penting dengan fakta grounding,
    // sehingga halusinasi yang gramatikal tetapi melenceng (mis. "turnip ... cucumber")
    // ikut tertolak. Ini memaksa keluaran model menjadi parafrase yang setia.
    const seedWords = new Set(seedContext.toLowerCase().match(/\b[a-z]{5,}\b/g) ?? []);
    const overlap = uniqueWords.filter((w) => seedWords.has(w)).length;

    const isFaithfulParaphrase =
      mentionsName && endsCleanly && text.length >= 25 && text.length <= 220 &&
      maxRepeat <= 3 && overlap >= 3;

    // Keluaran model hanya ditampilkan bila lolos semua pemeriksaan; jika tidak, gunakan
    // fakta terkurasi yang dirotasi (dijamin benar, relevan, dan tetap bervariasi).
    return isFaithfulParaphrase ? text : seedContext;
  }

  isReady() {
    return this.isModelLoaded;
  }
}