<script setup lang="ts">
import { ref } from 'vue'
import { useGenerationStore } from '@/stores/generation'

const props = defineProps<{
  mode: 'txt2img' | 'img2img'
}>()

const store = useGenerationStore()

const n = () => {
  if (!store.prompt || store.isLoading) return
  store.generateImage({
    prompt: store.prompt,
    negative_prompt: store.negativePrompt,
    steps: store.steps,
    seed: store.seed,
    cfgScale: store.cfgScale,
    strength: store.strength,
    sampler: store.sampler,
    width: store.width,
    height: store.height,
    saveImages: store.saveImages,
    initImage: props.mode === 'img2img' ? store.initImage : null
  })
}

const onFileChange = (e: Event) => {
  const target = e.target as HTMLInputElement
  if (target.files && target.files[0]) {
    const reader = new FileReader()
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string
      store.initImage = dataUrl
      
      // Get image dimensions
      const img = new Image()
      img.onload = () => {
        uploadedImageWidth.value = img.width
        uploadedImageHeight.value = img.height
      }
      img.src = dataUrl
    }
    reader.readAsDataURL(target.files[0])
  }
}

const uploadedImageWidth = ref(0)
const uploadedImageHeight = ref(0)

const useImageSize = () => {
  if (uploadedImageWidth.value > 0 && uploadedImageHeight.value > 0) {
    // Round to nearest multiple of 64
    store.width = Math.round(uploadedImageWidth.value / 64) * 64
    store.height = Math.round(uploadedImageHeight.value / 64) * 64
    
    // Ensure minimum of 64
    if (store.width < 64) store.width = 64
    if (store.height < 64) store.height = 64
  }
}

const clearInitImage = () => {
  store.initImage = null
  uploadedImageWidth.value = 0
  uploadedImageHeight.value = 0
}
</script>

<template>
  <div class="card shadow-sm p-3">
    <h5 class="card-title mb-4">
      {{ mode === 'txt2img' ? 'Text-to-Image' : 'Image-to-Image' }}
    </h5>
    <form @submit.prevent="n">
      <!-- Img2Img Upload -->
      <div class="mb-3" v-if="mode === 'img2img'">
        <label class="form-label">Initial Image:</label>
        <div v-if="!store.initImage" class="image-upload-dropzone border rounded p-4 text-center" @click="$refs.fileInput.click()">
          <i class="bi bi-cloud-arrow-up display-6"></i>
          <p class="mb-0 mt-2">Click to upload or drag & drop</p>
          <input type="file" ref="fileInput" class="d-none" accept="image/*" @change="onFileChange" />
        </div>
        <div v-else class="position-relative border rounded p-2 text-center">
          <img :src="store.initImage" class="img-thumbnail" style="max-height: 200px;" />
          <div class="mt-2 d-flex justify-content-center gap-2">
            <button type="button" class="btn btn-outline-secondary btn-sm" @click="useImageSize">
              <i class="bi bi-aspect-ratio"></i> Use Size ({{ uploadedImageWidth }}x{{ uploadedImageHeight }})
            </button>
            <button type="button" class="btn btn-danger btn-sm" @click="clearInitImage">
              <i class="bi bi-trash"></i> Clear
            </button>
          </div>
        </div>
      </div>

      <!-- Strength Slider (only if initImage exists) -->
      <div class="mb-3" v-if="mode === 'img2img' && store.initImage">
        <label for="strength" class="form-label d-flex justify-content-between">
          <span>Denoising Strength:</span>
          <span class="badge bg-primary">{{ store.strength }}</span>
        </label>
        <input type="range" class="form-range" id="strength" v-model.number="store.strength" min="0" max="1" step="0.01">
        <div class="form-text text-muted small">How much to change the initial image. 1.0 = completely new image.</div>
      </div>

      <div class="mb-3">
        <label for="prompt" class="form-label">Prompt:</label>
        <textarea
          id="prompt"
          v-model="store.prompt"
          rows="4"
          class="form-control"
          placeholder="A cinematic photograph of..."
          required
        ></textarea>
      </div>

      <div class="mb-3">
        <label for="negativePrompt" class="form-label">Negative Prompt:</label>
        <textarea
          id="negativePrompt"
          v-model="store.negativePrompt"
          rows="3"
          class="form-control"
          placeholder="deformed, bad anatomy, blurry..."
        ></textarea>
      </div>

      <div class="row g-3 mb-3">
        <div class="col-md-6">
          <label for="steps" class="form-label">Steps:</label>
          <input
            type="number"
            id="steps"
            v-model.number="store.steps"
            min="1"
            max="150"
            class="form-control"
          />
        </div>
        <div class="col-md-6">
          <label for="seed" class="form-label">Seed:</label>
          <input
            type="number"
            id="seed"
            v-model.number="store.seed"
            class="form-control"
          />
        </div>
      </div>

      <div class="row g-3 mb-3">
        <div class="col-md-6">
          <label for="width" class="form-label">Width:</label>
          <input
            type="number"
            id="width"
            v-model.number="store.width"
            min="64"
            max="2048"
            step="64"
            class="form-control"
          />
        </div>
        <div class="col-md-6">
          <label for="height" class="form-label">Height:</label>
          <input
            type="number"
            id="height"
            v-model.number="store.height"
            min="64"
            max="2048"
            step="64"
            class="form-control"
          />
        </div>
      </div>

      <div class="row g-3 mb-4">
        <div class="col-md-6">
          <label for="cfgScale" class="form-label">CFG Scale:</label>
          <input
            type="number"
            id="cfgScale"
            v-model.number="store.cfgScale"
            min="1"
            max="30"
            step="0.5"
            class="form-control"
          />
        </div>
        <div class="col-md-6">
          <label for="sampler" class="form-label">Sampler:</label>
          <select
            id="sampler"
            v-model="store.sampler"
            class="form-select"
          >
            <option v-for="s in store.samplers" :key="s" :value="s">{{ s }}</option>
          </select>
        </div>
      </div>

      <div class="d-grid">
        <button
          type="submit"
          class="btn btn-primary"
          :disabled="store.isLoading"
        >
          <span v-if="store.isLoading" class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
          {{ store.isLoading ? ' Generating...' : 'Generate' }}
        </button>
      </div>
    </form>
  </div>
</template>

<style scoped>
.image-upload-dropzone {
  cursor: pointer;
  transition: all 0.2s;
  background-color: #f8f9fa;
  color: #6c757d;
}

[data-bs-theme="dark"] .image-upload-dropzone {
  background-color: #2b3035 !important;
  color: #adb5bd !important;
  border-color: #495057 !important;
}

.image-upload-dropzone:hover {
  background-color: #e9ecef;
  border-color: #0d6efd !important;
}

[data-bs-theme="dark"] .image-upload-dropzone:hover {
  background-color: #373b3e !important;
  border-color: #3d8bfd !important;
}
</style>
