<script setup lang="ts">
import ImageGallery from '../components/ImageGallery.vue'
import { ref } from 'vue'

const galleryRef = ref<any>(null)
</script>

<template>
  <div class="history-view h-100 d-flex flex-column">
    <div class="row mb-3 align-items-center">
      <div class="col">
        <h3 class="mb-0">Image History</h3>
      </div>
      <div class="col-auto d-flex align-items-center gap-3 bg-body-secondary p-2 px-3 rounded shadow-sm">
        <div class="d-flex align-items-center gap-2">
          <span class="small text-muted text-uppercase fw-bold text-nowrap" style="font-size: 0.7rem;">Images per row: {{ galleryRef?.columnsPerRow }}</span>
          <input 
            type="range" 
            class="form-range" 
            style="width: 150px;"
            min="2" 
            max="12" 
            step="1"
            :value="galleryRef?.columnsPerRow"
            @input="galleryRef?.setColumns(parseInt(($event.target as HTMLInputElement).value))"
          >
        </div>
      </div>
    </div>
    <hr class="mt-0 mb-4 opacity-10">
    <div class="flex-grow-1 overflow-auto">
      <ImageGallery ref="galleryRef" />
    </div>
  </div>
</template>

<style scoped>
.form-range {
  height: 1.5rem;
}

/* Ensure the range track is visible in both modes */
.form-range::-webkit-slider-runnable-track {
  background-color: var(--bs-border-color);
  height: 0.4rem;
  border-radius: 1rem;
}

.form-range::-moz-range-track {
  background-color: var(--bs-border-color);
  height: 0.4rem;
  border-radius: 1rem;
}

.form-range::-webkit-slider-thumb {
  margin-top: -0.3rem; /* Center the thumb on the track */
}
</style>
