<script setup lang="ts">
import { useGenerationStore } from '@/stores/generation'

const store = useGenerationStore()
</script>

<template>
  <div>
    <h5 class="card-title mb-4">Result</h5>
    <div class="image-display-container">
      <div v-if="store.isLoading" class="d-flex flex-column align-items-center justify-content-center h-100 text-muted">
        <div class="spinner-border text-primary" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-3">Generating image...</p>
      </div>
      <div v-else-if="store.error" class="alert alert-danger h-100">
        <h4 class="alert-heading">Error</h4>
        <p>{{ store.error }}</p>
      </div>
      <div v-else-if="store.imageUrl">
        <img :src="store.imageUrl" alt="Generated Image" class="img-fluid rounded" />
      </div>
      <div v-else class="d-flex align-items-center justify-content-center h-100 text-muted">
        <p>No image generated yet.</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.image-display-container {
  min-height: 400px; /* Ensure a consistent height for the container */
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
