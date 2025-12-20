<script setup lang="ts">
import { onMounted } from 'vue'
import { useGenerationStore } from '@/stores/generation'

const store = useGenerationStore()

onMounted(() => {
  store.fetchModels()
})

const handleModelChange = (event: Event) => {
  const target = event.target as HTMLSelectElement
  if (target.value) {
    store.loadModel(target.value)
  }
}
</script>

<template>
  <div class="p-3">
    <h5 class="mb-3">Model</h5>
    <div class="mb-4">
      <select 
        class="form-select form-select-sm" 
        :value="store.currentModel" 
        @change="handleModelChange"
        :disabled="store.isModelSwitching || store.isGenerating"
      >
        <option v-if="store.isModelsLoading" disabled>Loading models...</option>
        <template v-else>
          <!-- Only show main models from stable-diffusion directory for a cleaner UI -->
          <option v-for="model in store.models.filter(m => m.type === 'stable-diffusion' || m.type === 'root')" :key="model.id" :value="model.id">
            {{ model.name }}
          </option>
        </template>
      </select>
      <div v-if="store.isModelSwitching && !store.isModelsLoading" class="mt-2 small text-primary">
         <span class="spinner-border spinner-border-sm"></span> Switching model...
      </div>
    </div>

    <h5 class="mb-3">Menu</h5>
    <ul class="nav nav-pills flex-column">
      <li class="nav-item">
        <router-link to="/" class="nav-link">Text-to-Image</router-link>
      </li>
      <li class="nav-item">
        <router-link to="/img2img" class="nav-link">Image-to-Image</router-link>
      </li>
      <li class="nav-item">
        <router-link to="/exploration" class="nav-link">Dynamic Exploration</router-link>
      </li>
      <li class="nav-item">
        <router-link to="/settings" class="nav-link">Settings</router-link>
      </li>
      <li class="nav-item">
        <router-link to="/history" class="nav-link">History</router-link>
      </li>
      <!-- Add other nav-items for new views here in the future -->
    </ul>
  </div>
</template>

<style scoped>
/* Scoped styles for the sidebar */
.nav-link {
  cursor: pointer;
}

/* vue-router automatically adds this class to the active link */
.router-link-active {
  background-color: var(--bs-primary);
  color: var(--bs-nav-pills-link-active-color);
}
</style>
