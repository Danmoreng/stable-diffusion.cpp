import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

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
  }

  // Load state from localStorage or use defaults
  const savedSettings = localStorage.getItem('webui-settings')
  const initialState = savedSettings ? { ...defaults, ...JSON.parse(savedSettings) } : defaults

  const isLoading = ref(false)
  const imageUrl = ref<string | null>(null)
  const error = ref<string | null>(null)

  // State for parameters
  const prompt = ref(initialState.prompt)
  const negativePrompt = ref(initialState.negativePrompt)
  const steps = ref(initialState.steps)
  const seed = ref(-1) // Seed is not persisted
  const cfgScale = ref(initialState.cfgScale)
  const strength = ref(initialState.strength)
  const sampler = ref(initialState.sampler)
  const samplers = ref(['euler', 'euler_a', 'heun', 'dpm2', 'dpmpp_2s_a', 'dpmpp_2m', 'dpmpp_2mv2', 'ipndm', 'ipndm_v', 'lcm', 'ddim_trailing', 'tcd'])
  const width = ref(initialState.width)
  const height = ref(initialState.height)

  // Img2Img State
  const initImage = ref<string | null>(null)

  // UI State
  const isSidebarCollapsed = ref(false)
  const theme = ref(initialState.theme)
  const saveImages = ref(initialState.saveImages)

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
  watch([prompt, negativePrompt, steps, cfgScale, sampler, width, height, theme, saveImages, strength], (newValues) => {
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
    sampler: string
    width: number
    height: number
    saveImages: boolean
    initImage?: string | null
  }

  async function generateImage(params: GenerationParams) {
    isLoading.value = true
    imageUrl.value = null
    error.value = null

    try {
      const body: any = {
        prompt: params.prompt,
        negative_prompt: params.negative_prompt,
        sample_steps: params.steps,
        cfg_scale: params.cfgScale,
        strength: params.strength,
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
      })

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || 'An error occurred while generating the image.')
      }

      const responseData = await response.json();
      const b64Json = responseData.data[0].b64_json;
      if (!b64Json) {
        throw new Error('Server response did not contain image data.');
      }
      imageUrl.value = `data:image/png;base64,${b64Json}`;

    } catch (e: any) {
      error.value = e.message
      console.error(e)
    } finally {
      isLoading.value = false
    }
  }

  return { isLoading, imageUrl, error, generateImage, prompt, negativePrompt, steps, seed, cfgScale, strength, sampler, samplers, width, height, isSidebarCollapsed, toggleSidebar, theme, toggleTheme, saveImages, initImage }
})
