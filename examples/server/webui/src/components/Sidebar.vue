<script setup lang="ts">
// This component is now only for navigation
</script>

<template>
  <div class="p-3">
    <h5 class="mb-3">Menu</h5>
    <ul class="nav nav-pills flex-column">
      <li class="nav-item">
        <router-link to="/" class="nav-link">Text-to-Image</router-link>
      </li>
      <li class="nav-item">
        <router-link to="/img2img" class="nav-link">Image-to-Image</router-link>
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
