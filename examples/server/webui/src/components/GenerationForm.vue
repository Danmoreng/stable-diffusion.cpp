<script setup lang="ts">
import { useGenerationStore } from '@/stores/generation'

const store = useGenerationStore()

const generateImage = () => {
  if (!store.prompt || store.isLoading) {
    return
  }
  // Pass all relevant params from the store
  store.generateImage({
    prompt: store.prompt,
    negative_prompt: store.negativePrompt,
    steps: store.steps,
    seed: store.seed,
    cfgScale: store.cfgScale,
    sampler: store.sampler,
    width: store.width,
    height: store.height,
    saveImages: store.saveImages,
  })
}
</script>

<template>
  <div>
    <h5 class="card-title mb-4">Generation Parameters</h5>
    <form @submit.prevent="generateImage">
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
/* Add any component-specific styles here if needed */
</style>
