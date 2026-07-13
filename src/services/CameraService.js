export class CameraService {
  constructor() {
    this.stream = null;
    this.video = null;
    this.canvas = null;
  }

  setVideoElement(videoElement) {
    this.video = videoElement;
  }

  setCanvasElement(canvasElement) {
    this.canvas = canvasElement;
  }

  async loadCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === 'videoinput');
  }

  async startCamera(cameraType = 'default') {
    this.stopCamera();

    const constraints = {
      video: {
        facingMode: cameraType === 'front' ? 'user' : 'environment',
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (this.video) {
        this.video.srcObject = this.stream;

        // Penting untuk iOS: Pastikan playsinline, muted, dan autoplay diatur
        this.video.setAttribute('playsinline', '');
        this.video.setAttribute('muted', '');
        this.video.setAttribute('autoplay', '');

        // Tunggu video benar-benar siap
        await new Promise((resolve) => {
          this.video.onloadedmetadata = () => {
            this.video.play().then(resolve);
          };
        });
      }
      return this.stream;
    } catch (error) {
      console.error('Error starting camera:', error);
      throw error;
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
  }

  isActive() {
    return !!this.stream;
  }

  isReady() {
    return this.video && this.video.readyState === 4;
  }
}