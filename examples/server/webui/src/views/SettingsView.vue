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
          <label for="themeSelect" class="form-label mb-0">Theme</label>
          <select id="themeSelect" class="form-select form-select-sm" v-model="store.theme" style="width: auto; min-width: 120px;">
            <option value="system">🖥️ System</option>
            <option value="light">☀️ Light</option>
            <option value="dark">🌙 Dark</option>
          </select>
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
