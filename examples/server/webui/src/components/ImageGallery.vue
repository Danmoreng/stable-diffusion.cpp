<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import { Modal, Carousel } from 'bootstrap'

const images = ref<string[]>([])
const isLoading = ref(true)
const error = ref<string | null>(null)

const modalElement = ref<HTMLElement | null>(null)
const carouselElement = ref<HTMLElement | null>(null)
let modalInstance: Modal | null = null
let carouselInstance: Carousel | null = null

async function fetchImages() {
  isLoading.value = true
  error.value = null
  try {
    const response = await fetch('/v1/history/images')
    if (!response.ok) {
      throw new Error('Failed to fetch image history from the server.')
    }
    const data = await response.json()
    images.value = data

    // Wait for the DOM to update with the new images
    await nextTick()
    
    // Initialize modal and carousel now that the elements are in the DOM
    if (modalElement.value && carouselElement.value) {
      modalInstance = new Modal(modalElement.value)
      carouselInstance = new Carousel(carouselElement.value, {
        interval: false, // Do not auto-cycle
        touch: true,
      })
    }

  } catch (e: any) {
    error.value = e.message
  } finally {
    isLoading.value = false
  }
}

function openModal(index: number) {
  if (carouselInstance && modalInstance) {
    carouselInstance.to(index)
    modalInstance.show()
  }
}

onMounted(() => {
  fetchImages()
})
</script>

<template>
  <div>
    <!-- Loading/Error/Empty State -->
    <div v-if="isLoading" class="text-center">
      <div class="spinner-border" role="status">
        <span class="visually-hidden">Loading...</span>
      </div>
    </div>
    <div v-else-if="error" class="alert alert-danger">
      {{ error }}
    </div>
    <div v-else-if="images.length === 0" class="text-center text-muted">
      <p>No images found in history.</p>
      <p>Generate some images with the "Save Images Automatically" setting enabled.</p>
    </div>

    <!-- Image Grid -->
    <div v-else class="row g-3">
      <div v-for="(image, index) in images" :key="image" class="col-xl-3 col-lg-4 col-md-6">
        <div class="card card-clickable" @click="openModal(index)">
          <img :src="'/outputs/' + image" class="card-img-top" :alt="image" />
        </div>
      </div>
    </div>

    <!-- Modal -->
    <div class="modal fade" ref="modalElement" tabindex="-1" aria-labelledby="imageModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-xl modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="imageModalLabel">Image Viewer</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body p-0">
            <!-- Carousel -->
            <div ref="carouselElement" id="imageHistoryCarousel" class="carousel slide">
              <div class="carousel-inner">
                <div v-for="(image, index) in images" :key="`carousel-${image}`" class="carousel-item" :class="{ active: index === 0 }">
                  <img :src="'/outputs/' + image" class="d-block w-100" :alt="image">
                </div>
              </div>
              <button class="carousel-control-prev" type="button" data-bs-target="#imageHistoryCarousel" data-bs-slide="prev">
                <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                <span class="visually-hidden">Previous</span>
              </button>
              <button class="carousel-control-next" type="button" data-bs-target="#imageHistoryCarousel" data-bs-slide="next">
                <span class="carousel-control-next-icon" aria-hidden="true"></span>
                <span class="visually-hidden">Next</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

  </div>
</template>

<style scoped>
.card-img-top {
  aspect-ratio: 1 / 1;
  object-fit: cover;
}
.card-clickable {
  cursor: pointer;
  transition: transform 0.2s ease-in-out;
}
.card-clickable:hover {
  transform: scale(1.03);
}
.modal-body img {
  max-height: 80vh;
  object-fit: contain;
}
</style>
