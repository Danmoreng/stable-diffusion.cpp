import { defineStore } from 'pinia'
import { ref, watch, computed } from 'vue'

export const useGenerationStore = defineStore('generation', () => {
  // --- State Initialization ---
  const defaults = {
    prompt: `A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt.`,
    negativePrompt: 'deformed, bad anatomy, bad proportions, blurry, cloned face, cropped, gross proportions, jpeg artifacts, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, username, watermark, signature',
    steps: 4,
    cfgScale: 1.0,
    sampler: 'euler_a',
    width: 1024,
    height: 768,
    theme: 'dark' as 'light' | 'dark',
    saveImages: true,
    strength: 0.75,
    batchCount: 1,
  }

  // Load state from localStorage or use defaults
  const savedSettings = localStorage.getItem('webui-settings')
  const initialState = savedSettings ? { ...defaults, ...JSON.parse(savedSettings) } : defaults

  const isGenerating = ref(false)
  const isModelSwitching = ref(false)
  const imageUrls = ref<string[]>([])
  const error = ref<string | null>(null)

  // State for parameters
  const prompt = ref(initialState.prompt)
  const negativePrompt = ref(initialState.negativePrompt)
  const steps = ref(initialState.steps)
  const seed = ref(-1) // Seed is not persisted
  const cfgScale = ref(initialState.cfgScale)
  const strength = ref(initialState.strength)
  const batchCount = ref(initialState.batchCount)
  const sampler = ref(initialState.sampler)
  const samplers = ref(['euler', 'euler_a', 'heun', 'dpm2', 'dpmpp_2s_a', 'dpmpp_2m', 'dpmpp_2mv2', 'ipndm', 'ipndm_v', 'lcm', 'ddim_trailing', 'tcd'])
  const width = ref(initialState.width)
  const height = ref(initialState.height)

  // Img2Img State
  const initImage = ref<string | null>(null)

  // UI State
  const isSidebarCollapsed = ref(localStorage.getItem('sidebar-collapsed') === 'true')
  const theme = ref(initialState.theme)
  const saveImages = ref(initialState.saveImages)

  // Model Management State
  const models = ref<any[]>([])
  const currentModel = ref<string>('')
  const isModelsLoading = ref(false)

  // Watch for sidebar changes and persist
  watch(isSidebarCollapsed, (newVal) => {
    localStorage.setItem('sidebar-collapsed', String(newVal))
  })

  // Progress State
  const progressStep = ref(0)
  const progressSteps = ref(0)
  const progressTime = ref(0)
  const progressPhase = ref('')
  let progressSource: EventSource | null = null
  
  // Helpers for better ETA
  const lastStepTime = ref(0);
  const lastStepIndex = ref(0);
  const stepTimeHistory = ref<number[]>([]);

  const eta = computed(() => {
    if (progressSteps.value === 0 || progressStep.value === 0) return 0;
    
    // Use the average of the last few steps if available for a more stable estimate
    // otherwise fallback to total average
    const history = stepTimeHistory.value;
    const avgStepTime = history.length > 0 
      ? history.reduce((a, b) => a + b, 0) / history.length 
      : progressTime.value / progressStep.value;

    const remainingSteps = progressSteps.value - progressStep.value;
    return Math.round(avgStepTime * remainingSteps);
  });

  function startStreamingProgress() {
    if (progressSource) progressSource.close();
    progressStep.value = 0;
    progressSteps.value = 0;
    progressTime.value = 0;
    progressPhase.value = 'Initializing...';
    lastStepTime.value = 0;
    lastStepIndex.value = 0;
    stepTimeHistory.value = [];
    
    console.log('Starting progress stream...');
    progressSource = new EventSource('/v1/stream/progress');
    
    progressSource.onopen = () => {
      console.log('Progress stream connection opened.');
    };

    progressSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.phase) {
          if (data.phase !== progressPhase.value) {
            // Reset history on phase change as VAE steps have different timings than Sampling
            stepTimeHistory.value = [];
          }
          progressPhase.value = data.phase;
        }

        // Calculate time for this specific step
        if (data.step > lastStepIndex.value) {
          const deltaT = data.time - lastStepTime.value;
          const deltaS = data.step - lastStepIndex.value;
          const timePerStep = deltaT / deltaS;
          
          if (timePerStep > 0) {
            stepTimeHistory.value.push(timePerStep);
            // Keep only the last 5 steps for a rolling average
            if (stepTimeHistory.value.length > 5) {
              stepTimeHistory.value.shift();
            }
          }
          
          lastStepTime.value = data.time;
          lastStepIndex.value = data.step;
        } else if (data.step < lastStepIndex.value) {
          // New sub-batch or reset
          lastStepTime.value = data.time;
          lastStepIndex.value = data.step;
        }

        progressStep.value = data.step;
        progressSteps.value = data.steps;
        progressTime.value = data.time;
      } catch (e) {
        console.error('Error parsing progress stream data:', event.data, e);
      }
    };

    progressSource.onerror = (err) => {
      console.error('Progress stream error:', err);
    };
  }

  function stopStreamingProgress() {
    if (progressSource) {
      progressSource.close();
      progressSource = null;
    }
    progressStep.value = 0;
    progressSteps.value = 0;
    progressTime.value = 0;
    progressPhase.value = '';
  }

  async function fetchModels() {
    isModelsLoading.value = true
    try {
      const response = await fetch('/v1/models')
      const data = await response.json()
      models.value = data.data
      const activeModel = models.value.find(m => m.active)
      if (activeModel) {
        currentModel.value = activeModel.id
      }
    } catch (e) {
      console.error('Failed to fetch models:', e)
    } finally {
      isModelsLoading.value = false
    }
  }

  async function loadModel(modelId: string) {
    isModelSwitching.value = true
    try {
      const response = await fetch('/v1/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      })
      if (!response.ok) throw new Error('Failed to load model')
      currentModel.value = modelId
      // Refresh models list to update active status
      await fetchModels()
    } catch (e: any) {
      error.value = e.message
    } finally {
      isModelSwitching.value = false
    }
  }

  function toggleSidebar() {
    isSidebarCollapsed.value = !isSidebarCollapsed.value
  }

  function toggleTheme() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark';
  }

  // --- Effects ---

  // Watch for theme changes and apply them to the root element
  watch(theme, (newTheme) => {
    document.documentElement.setAttribute('data-bs-theme', newTheme)
  }, { immediate: true })


  // Watch for changes in settings and persist them to localStorage
  watch([prompt, negativePrompt, steps, cfgScale, sampler, width, height, theme, saveImages, strength, batchCount], (newValues) => {
    const settingsToSave = {
      prompt: newValues[0],
      negativePrompt: newValues[1],
      steps: newValues[2],
      cfgScale: newValues[3],
      sampler: newValues[4],
      width: newValues[5],
      height: newValues[6],
      theme: newValues[7],
      saveImages: newValues[8],
      strength: newValues[9],
      batchCount: newValues[10],
    }
    localStorage.setItem('webui-settings', JSON.stringify(settingsToSave))
  }, { deep: true })


  // --- Actions ---

  interface GenerationParams {
    prompt: string
    negative_prompt: string
    steps: number
    seed: number
    cfgScale: number
    strength: number
    batchCount: number
    sampler: string
    width: number
    height: number
    saveImages: boolean
    initImage?: string | null
  }

  async function requestImage(params: GenerationParams, signal?: AbortSignal): Promise<string[]> {
    const body: any = {
      prompt: params.prompt,
      negative_prompt: params.negative_prompt,
      sample_steps: params.steps,
      cfg_scale: params.cfgScale,
      strength: params.strength,
      n: params.batchCount,
      sampling_method: params.sampler.toLowerCase().replace(' a', '_a').replace(/\+\+/g, 'pp'),
      seed: params.seed,
      width: params.width,
      height: params.height,
      save_image: params.saveImages,
    }

    if (params.initImage) {
      body.init_image = params.initImage
    }

    const response = await fetch('/v1/images/generations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal,
    })

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(errText || 'An error occurred while generating the image.')
    }

    const responseData = await response.json();
    if (!responseData.data || responseData.data.length === 0) {
      throw new Error('Server response did not contain image data.');
    }
    
    return responseData.data.map((item: any) => `data:image/png;base64,${item.b64_json}`);
  }

  async function generateImage(params: GenerationParams) {
    if (isModelSwitching.value) return;
    isGenerating.value = true
    imageUrls.value = []
    error.value = null
    startStreamingProgress();

    try {
      imageUrls.value = await requestImage(params);
    } catch (e: any) {
      error.value = e.message
      console.error(e)
    } finally {
      isGenerating.value = false
      stopStreamingProgress();
    }
  }

  return { isGenerating, isModelSwitching, imageUrls, error, generateImage, requestImage, prompt, negativePrompt, steps, seed, cfgScale, strength, batchCount, sampler, samplers, width, height, isSidebarCollapsed, toggleSidebar, theme, toggleTheme, saveImages, initImage, models, currentModel, isModelsLoading, fetchModels, loadModel, progressStep, progressSteps, progressTime, progressPhase, eta, startStreamingProgress, stopStreamingProgress }
})
