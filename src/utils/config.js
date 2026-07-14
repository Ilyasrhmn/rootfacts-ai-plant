export const APP_CONFIG = {
  detectionConfidenceThreshold: 85,
  analyzingDelay: 2000,
  factsGenerationDelay: 2000,
  detectionRetryInterval: 100
};

export const TONE_CONFIG = {
  availableTones: [
    { value: 'normal', label: 'Normal' },
    { value: 'funny', label: 'Lucu' },
    { value: 'professional', label: 'Profesional' },
    { value: 'casual', label: 'Santai' }
  ],
  defaultTone: 'normal'
};

export const MODEL_CONFIG = {
  detectionModel: '/model/model.json',
  detectionMetadata: '/model/metadata.json',
  // Model instruction-tuned yang jauh lebih baik daripada flan-t5-small untuk
  // menghasilkan fun fact yang koheren, bervariasi, dan tidak mengorupsi nama sayuran.
  transformersModel: 'Xenova/LaMini-Flan-T5-248M'
};

export const isValidDetection = (result) => {
  const { detectionConfidenceThreshold } = APP_CONFIG;
  return result && result.isValid && result.confidence >= detectionConfidenceThreshold;
};

/**
 * Validasi response fetch untuk memastikan bukan HTML (404 redirect)
 * Mencegah error "Unexpected token <" yang membingungkan.
 */
export const validateResponse = async (response) => {
  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('text/html')) {
    throw new Error(`Model File Not Found (404): Expected JSON/Binary but got HTML. Check path: ${response.url}`);
  }
  if (!response.ok) {
    throw new Error(`Network response was not ok: ${response.statusText}`);
  }
  return response;
};
