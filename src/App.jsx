import { useEffect, useRef, useCallback, useState } from 'react';
import Header from './components/Header';
import CameraSection from './components/CameraSection';
import InfoPanel from './components/InfoPanel';
import { useAppState } from './hooks/useAppState';
import { DetectionService } from './services/DetectionService';
import { RootFactsService } from './services/RootFactsService';
import { CameraService } from './services/CameraService';
import { APP_CONFIG } from './utils/config';

function App() {
  const { state, actions } = useAppState();
  const [currentTone, setCurrentTone] = useState('normal');
  const [cameraType, setCameraType] = useState('default');

  const animationFrameRef = useRef(null);
  const lastPredictionTimeRef = useRef(0);
  const fpsLimit = 500; // Pembatasan 2 FPS untuk efisiensi CPU/GPU

  // Inisialisasi layanan AI saat komponen dimuat
  useEffect(() => {
    const detector = new DetectionService();
    const generator = new RootFactsService();
    const camera = new CameraService();

    actions.setServices({ detector, generator, camera });

    const initModels = async () => {
      try {
        actions.setModelStatus('Menginisialisasi Si Mata (Vision)...');
        await detector.loadModel((p) => actions.setModelStatus(`Memuat Si Mata: ${p}%`));

        actions.setModelStatus('Menginisialisasi Si Otak (NLP)...');
        await generator.loadModel((p) => actions.setModelStatus(`Memuat Si Otak: ${p}%`));

        actions.setModelStatus('Model AI Siap');
      } catch (err) {
        actions.setError('Gagal memuat model AI. Periksa koneksi internet Anda.');
        console.error(err);
      }
    };

    initModels();

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      detector.dispose();
      camera.stopCamera();
    };
  }, [actions]);

  // Loop deteksi dengan batasan FPS
  const runDetection = useCallback(async () => {
    if (!state.isRunning) return;

    const now = performance.now();
    if (now - lastPredictionTimeRef.current >= fpsLimit) {
      if (state.services.camera && state.services.camera.isReady()) {
        const result = await state.services.detector.predict(state.services.camera.video);

        if (result && result.score > (APP_CONFIG.detectionConfidenceThreshold / 100)) {
          actions.setDetectionResult(result);
          actions.setAppState('result');

          state.services.camera.stopCamera();
          actions.setRunning(false);

          handleGenerateFact(result.className);
          return;
        }
      }
      lastPredictionTimeRef.current = now;
    }

    animationFrameRef.current = requestAnimationFrame(runDetection);
  }, [state.isRunning, state.services, actions]);

  useEffect(() => {
    if (state.isRunning) {
      runDetection();
    } else {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    }
  }, [state.isRunning, runDetection]);

  const handleToggleCamera = async () => {
    if (!state.services.camera) return;

    if (state.isRunning) {
      state.services.camera.stopCamera();
      actions.setRunning(false);
    } else {
      // Feedback jika model belum siap
      if (state.modelStatus !== 'Model AI Siap') {
        alert('Mohon tunggu sebentar, model AI sedang disiapkan...');
        return;
      }

      try {
        actions.resetResults();
        await state.services.camera.startCamera(cameraType);
        actions.setRunning(true);
        actions.setAppState('analyzing');
      } catch (err) {
        actions.setError('Gagal mengakses kamera. Pastikan izin telah diberikan.');
        console.error(err);
      }
    }
  };

  const handleCameraTypeChange = (type) => {
    setCameraType(type);
  };

  const handleGenerateFact = async (label) => {
    actions.setFunFactData(null);
    try {
      const fact = await state.services.generator.generateFacts(label);
      actions.setFunFactData(fact);
    } catch (err) {
      actions.setFunFactData('error');
    }
  };

  const handleToneChange = (tone) => {
    setCurrentTone(tone);
    if (state.services.generator) {
      state.services.generator.setTone(tone);
    }
  };

  // 3. Copy to Clipboard Functionality
  const handleCopyFact = () => {
    if (state.funFactData && state.funFactData !== 'error') {
      navigator.clipboard.writeText(state.funFactData)
        .then(() => alert('Fakta berhasil disalin!'))
        .catch((err) => console.error('Gagal menyalin:', err));
    }
  };

  return (
    <div className="app-container">
      <Header modelStatus={state.modelStatus} />

      <main className="main-content">
        <CameraSection
          isRunning={state.isRunning}
          onToggleCamera={handleToggleCamera}
          onToneChange={handleToneChange}
          onCameraTypeChange={handleCameraTypeChange}
          services={state.services}
          modelStatus={state.modelStatus}
          error={state.error}
          currentTone={currentTone}
        />

        <InfoPanel
          appState={state.appState}
          detectionResult={state.detectionResult}
          funFactData={state.funFactData}
          error={state.error}
          onCopyFact={handleCopyFact}
        />
      </main>

      <footer className="footer">
        <p>&copy; 2026 Root Fact App - Advanced AI Submission</p>
      </footer>
    </div>
  );
}

export default App;
