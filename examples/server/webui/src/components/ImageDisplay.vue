<script setup lang="ts">
import { useGenerationStore } from '@/stores/generation'

const store = useGenerationStore()
</script>

<template>
  <div>
    <h5 class="card-title mb-4">Result</h5>
    <div class="image-display-container">
      <!-- Loading State -->
      <div v-if="store.isLoading" class="d-flex flex-column align-items-center justify-content-center h-100 text-muted p-5">
        <div class="spinner-border text-primary" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-3">Generating image(s)...</p>
      </div>

      <!-- Error State -->
      <div v-else-if="store.error" class="alert alert-danger h-100">
        <h4 class="alert-heading">Error</h4>
        <p>{{ store.error }}</p>
      </div>

      <!-- Results Grid -->
      <div v-else-if="store.imageUrls.length > 0" class="row g-2">
        <div 
          v-for="(url, index) in store.imageUrls" 
          :key="index" 
          :class="{
            'col-12': store.imageUrls.length === 1,
            'col-6': store.imageUrls.length > 1
          }"
        >
          <a :href="url" target="_blank">
            <img :src="url" :alt="'Generated Image ' + (index + 1)" class="img-fluid rounded shadow-sm" />
          </a>
        </div>
      </div>

      <!-- Empty State -->
      <div v-else class="d-flex align-items-center justify-content-center h-100 text-muted p-5">
        <p>No images generated yet.</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.image-display-container {
  min-height: 400px;
}
.img-fluid {
  width: 100%;
  height: auto;
  object-fit: contain;
}
</style>
