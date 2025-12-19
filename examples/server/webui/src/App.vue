<script setup lang="ts">
import Sidebar from './components/Sidebar.vue'
import { useGenerationStore } from '@/stores/generation'

const store = useGenerationStore()
</script>

<template>
  <div id="app" class="d-flex flex-column vh-100">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark flex-shrink-0">
      <div class="container-fluid">
        <button class="btn btn-outline-secondary me-2" @click="store.toggleSidebar">
          <span>&#9776;</span> <!-- Hamburger Icon -->
        </button>
        <a class="navbar-brand" href="#">Stable Diffusion WebUI</a>
      </div>
    </nav>

    <div id="content-wrapper" class="flex-grow-1" :class="{ 'sidebar-collapsed': store.isSidebarCollapsed }">
      <aside class="sidebar border-end">
        <Sidebar />
      </aside>
      <main class="main-content p-3">
        <router-view />
      </main>
    </div>
  </div>
</template>
