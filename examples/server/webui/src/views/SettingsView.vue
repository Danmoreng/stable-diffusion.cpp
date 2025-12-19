<script setup lang="ts">
import { useGenerationStore } from '@/stores/generation'
const store = useGenerationStore()
</script>

<template>
  <div>
    <h3>Settings</h3>
    <hr />
    <div class="row g-4">
      <div class="col-md-6">
        <h5>UI Settings</h5>
        <div class="d-flex justify-content-between align-items-center">
          <label for="themeToggle" class="form-label mb-0">Theme</label>
          <button id="themeToggle" class="btn btn-outline-secondary" @click="store.toggleTheme">
            <span v-if="store.theme === 'dark'">☀️ Light</span>
            <span v-else>🌙 Dark</span>
          </button>
        </div>
      </div>
      <div class="col-md-6">
        <h5>Image Generation</h5>
        <div class="form-check form-switch">
          <input class="form-check-input" type="checkbox" role="switch" id="saveImages" v-model="store.saveImages">
          <label class="form-check-label" for="saveImages">Save Images Automatically</label>
        </div>
      </div>
    </div>
  </div>
</template>
